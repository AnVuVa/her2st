from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List

import torch
from dataset import Her2stGenePredictionDataset
from model import AModel
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step-by-step performance probe for HER2ST training loop."
    )
    p.add_argument("--root_dir", type=str, default="data")
    p.add_argument("--hvg_path", type=str, default="data/her_hvg_cut_1000.npy")
    p.add_argument(
        "--sections", nargs="+", default=None, help="Optional list like A1 B1"
    )
    p.add_argument("--patch_size", type=int, default=112)
    p.add_argument("--selected_only", action="store_true", default=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pin_memory", action="store_true", default=False)
    return p.parse_args()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    s = sorted(vals)
    return {
        "mean": statistics.fmean(vals),
        "p50": s[int(0.50 * (len(s) - 1))],
        "p90": s[int(0.90 * (len(s) - 1))],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("[INFO] Building dataset...")
    t0 = time.perf_counter()
    ds = Her2stGenePredictionDataset(
        root_dir=args.root_dir,
        hvg_path=args.hvg_path,
        sections=args.sections,
        patch_size=args.patch_size,
        selected_only=args.selected_only,
    )
    t1 = time.perf_counter()
    print(
        f"[INFO] Dataset init: {t1 - t0:.3f}s, samples={len(ds)}, genes={len(ds.feature_genes)}"
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    model = AModel(out_dim=len(ds.feature_genes), base_channels=args.base_channels).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    fetch_times: List[float] = []
    h2d_times: List[float] = []
    fwd_times: List[float] = []
    bwd_times: List[float] = []
    opt_times: List[float] = []
    step_times: List[float] = []

    print("[INFO] Starting timed loop...")
    iterator = iter(dl)
    total_steps = args.warmup + args.steps

    for step in range(total_steps):
        s0 = time.perf_counter()

        a0 = time.perf_counter()
        batch = next(iterator, None)
        if batch is None:
            iterator = iter(dl)
            batch = next(iterator)
        a1 = time.perf_counter()

        sync_if_cuda(device)
        b0 = time.perf_counter()
        image = batch["image"].to(device, non_blocking=args.pin_memory)
        target = batch["target_gene"].to(device, non_blocking=args.pin_memory)
        sync_if_cuda(device)
        b1 = time.perf_counter()

        c0 = time.perf_counter()
        pred = model(image)
        loss = model.loss_fn(pred, target)
        sync_if_cuda(device)
        c1 = time.perf_counter()

        d0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync_if_cuda(device)
        d1 = time.perf_counter()

        e0 = time.perf_counter()
        opt.step()
        sync_if_cuda(device)
        e1 = time.perf_counter()

        s1 = time.perf_counter()

        if step >= args.warmup:
            fetch_times.append(a1 - a0)
            h2d_times.append(b1 - b0)
            fwd_times.append(c1 - c0)
            bwd_times.append(d1 - d0)
            opt_times.append(e1 - e0)
            step_times.append(s1 - s0)

        if step in (args.warmup, total_steps - 1):
            print(
                f"[STEP {step + 1}/{total_steps}] loss={loss.item():.5f} "
                f"fetch={a1 - a0:.3f}s h2d={b1 - b0:.3f}s fwd={c1 - c0:.3f}s bwd={d1 - d0:.3f}s opt={e1 - e0:.3f}s"
            )

    print("\n=== SUMMARY (seconds) ===")
    for name, values in [
        ("fetch", fetch_times),
        ("h2d", h2d_times),
        ("forward", fwd_times),
        ("backward", bwd_times),
        ("optim", opt_times),
        ("step_total", step_times),
    ]:
        s = summarize(values)
        print(f"{name:10s} mean={s['mean']:.4f} p50={s['p50']:.4f} p90={s['p90']:.4f}")

    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"\n[GPU] max_memory_allocated={max_mem:.1f} MiB")


def aggregate_results(path: str):
    import json
    import os

    files = os.listdir(path)
    result = 0
    total_samples = 0
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(path, file), "r") as f:
                data = json.load(f)
                result += data["mean_pcc"] * data["num_samples"]
                total_samples += data["num_samples"]
    print(f"PCC mean={result / total_samples} ({total_samples})")


if __name__ == "__main__":
    # main()
    aggregate_results("./_pipeline/results")

# python _pipeline/test.py --device cuda --steps 30 --warmup 5 --batch_size 16 --num_workers 4 --pin_memory

# === SUMMARY (seconds) ===
# fetch      mean=2.0005 p50=1.4582 p90=4.6902
# h2d        mean=0.0010 p50=0.0008 p90=0.0018
# forward    mean=0.0966 p50=0.0382 p90=0.1984
# backward   mean=0.0713 p50=0.0523 p90=0.1201
# optim      mean=0.0031 p50=0.0031 p90=0.0039
# step_total mean=2.1728 p50=1.5779 p90=4.9418

# [GPU] max_memory_allocated=567.5 MiB
