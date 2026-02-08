from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from dataset import Her2stGenePredictionDataset
from model import AModel
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_default_logger, log_stat, log_step


@dataclass
class PipelineConfig:
    root_dir: str = "../data"
    hvg_path: str = "../data/her_hvg_cut_1000.npy"
    patch_size: int = 112
    selected_only: bool = True

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20

    device: Optional[str] = "cuda"
    out_dir: str = "_pipeline/artifacts"
    cv_manifest_path: str = "_pipeline/cv_splits.yaml"


class Pipeline:
    def __init__(self, config: PipelineConfig, logger=None) -> None:
        self.config = config
        self.logger = logger or get_default_logger()

        self.device = torch.device(config.device)

        self.dataset: Optional[Her2stGenePredictionDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        self.model: Optional[AModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.history: List[Dict[str, float]] = []
        self.best_val_pcc: float = -float("inf")
        self.best_val_loss: float = float("inf")
        self.last_train_top20: List[Tuple[str, float]] = []
        self.last_val_top20: List[Tuple[str, float]] = []

    def build_data(self, fold: int) -> None:
        split = self._build_fold_split(fold)

        self.dataset = self._build_dataset(split["train"])
        val_ds = self._build_dataset(split["val"])
        test_ds = self._build_dataset(split["test"])

        if self.dataset.feature_genes != val_ds.feature_genes:
            raise RuntimeError("Train/val HVG feature mismatch.")
        if self.dataset.feature_genes != test_ds.feature_genes:
            raise RuntimeError("Train/test HVG feature mismatch.")

        self.train_loader = self._build_loader(self.dataset, shuffle=True)
        self.val_loader = self._build_loader(val_ds, shuffle=False)
        self.test_loader = self._build_loader(test_ds, shuffle=False)

        log_step(self.logger, f"build data for fold={fold}")
        log_stat(self.logger, "train_sections", len(split["train"]))
        log_stat(self.logger, "val_sections", len(split["val"]))
        log_stat(self.logger, "test_sections", len(split["test"]))
        log_stat(self.logger, "train_samples", len(self.dataset))
        log_stat(self.logger, "val_samples", len(val_ds))
        log_stat(self.logger, "test_samples", len(test_ds))

    def build_model(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Call build_data(fold=...) before build_model().")

        out_dim = len(self.dataset.feature_genes)
        self.model = AModel(out_dim=out_dim).to(self.device)
        log_step(self.logger, "build model")
        log_stat(self.logger, "model_out_dim", out_dim)

    def build_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Call build_model() before build_optimizer().")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        log_step(self.logger, "build optimizer")
        log_stat(self.logger, "lr", self.config.lr)

    def _build_dataset(self, sections: List[str]) -> Her2stGenePredictionDataset:
        return Her2stGenePredictionDataset(
            root_dir=self.config.root_dir,
            hvg_path=self.config.hvg_path,
            sections=sorted(sections),
            patch_size=self.config.patch_size,
            selected_only=self.config.selected_only,
            logger=self.logger,
        )

    def _build_loader(
        self, dataset: Her2stGenePredictionDataset, shuffle: bool
    ) -> DataLoader:
        pin = bool(self.config.pin_memory and self.device.type == "cuda")
        persistent = bool(
            self.config.persistent_workers and self.config.num_workers > 0
        )
        worker_kwargs = (
            {"prefetch_factor": self.config.prefetch_factor}
            if self.config.num_workers > 0
            else {}
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            **worker_kwargs,
        )

    def _build_fold_split(self, fold: int) -> Dict[str, List[str]]:
        manifest_path = Path(self.config.cv_manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"CV manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "splits" not in data or not isinstance(data["splits"], list):
            raise ValueError("Manifest must contain a 'splits' list.")

        row = None
        for item in data["splits"]:
            if int(item.get("fold", -1)) == int(fold):
                row = item
                break

        if row is None:
            raise ValueError(f"Fold {fold} not found in manifest.")

        train = [str(x) for x in row.get("train", [])]
        val = [str(x) for x in row.get("val", [])]
        test = [str(x) for x in row.get("test", [])]

        if not train or not val or not test:
            raise ValueError("Each fold must define non-empty train/val/test.")

        s_train, s_val, s_test = set(train), set(val), set(test)
        if (s_train & s_val) or (s_train & s_test) or (s_val & s_test):
            raise ValueError("train/val/test must be disjoint within a fold.")

        cnt_dir = Path(self.config.root_dir) / "ST-cnts"
        available = {p.stem for p in cnt_dir.glob("*.tsv")}
        missing = (s_train | s_val | s_test) - available
        if missing:
            raise ValueError(f"Sections not found in dataset: {sorted(missing)}")

        return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    def _build_pcc(
        self,
        pred_all: List[torch.Tensor],
        target_all: List[torch.Tensor],
        top_k: int = 20,
    ) -> Tuple[float, List[Tuple[str, float]]]:
        if not pred_all or not target_all:
            return float("nan"), []

        pred = torch.cat(pred_all, dim=0).float().numpy()
        target = torch.cat(target_all, dim=0).float().numpy()

        pcc_vals: List[float] = []
        for g in range(pred.shape[1]):
            try:
                r, _ = pearsonr(pred[:, g], target[:, g])
                pcc_vals.append(float(r) if np.isfinite(r) else float("nan"))
            except Exception:
                print(f"Warning: PCC computation failed for gene index {g}.")
                pcc_vals.append(float("nan"))

        pcc = np.asarray(pcc_vals, dtype=np.float64)
        finite = np.isfinite(pcc)
        mean_pcc = float(np.mean(pcc[finite])) if np.any(finite) else float("nan")

        if self.dataset is not None and hasattr(self.dataset, "feature_genes"):
            genes = self.dataset.feature_genes
        else:
            genes = [f"gene_{i}" for i in range(pcc.shape[0])]

        k = min(top_k, pcc.shape[0])
        order = np.argsort(np.nan_to_num(pcc, nan=-np.inf))[::-1][:k]
        top = [(genes[int(i)], float(pcc[int(i)])) for i in order]
        return mean_pcc, top

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        if self.model is None or self.optimizer is None or self.train_loader is None:
            raise RuntimeError(
                "Model, optimizer, and train_loader must be initialized."
            )

        self.model.train()
        total_loss = 0.0
        n_batches = 0
        pred_all: List[torch.Tensor] = []
        target_all: List[torch.Tensor] = []

        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            image = batch["image"].to(self.device)
            target = batch["target_gene"].to(self.device)

            pred = self.model(image)
            loss = self.model.loss_fn(pred, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1
            pred_all.append(pred.detach().cpu())
            target_all.append(target.detach().cpu())

        mean_loss = total_loss / max(1, n_batches)
        mean_pcc, top20 = self._build_pcc(pred_all, target_all)
        self.last_train_top20 = top20
        return {"epoch": float(epoch), "train_mse": mean_loss, "train_pcc": mean_pcc}

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int) -> Dict[str, float]:
        if self.model is None or self.val_loader is None:
            raise RuntimeError("Model and val_loader must be initialized.")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        pred_all: List[torch.Tensor] = []
        target_all: List[torch.Tensor] = []

        for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
            image = batch["image"].to(self.device)
            target = batch["target_gene"].to(self.device)
            pred = self.model(image)
            loss = self.model.loss_fn(pred, target)

            total_loss += float(loss.detach().item())
            n_batches += 1
            pred_all.append(pred.detach().cpu())
            target_all.append(target.detach().cpu())

        mean_loss = total_loss / max(1, n_batches)
        mean_pcc, top20 = self._build_pcc(pred_all, target_all)
        self.last_val_top20 = top20
        return {"epoch": float(epoch), "val_mse": mean_loss, "val_pcc": mean_pcc}

    def fit(self, fold: int, start_epoch: int = 1) -> List[Dict[str, float]]:
        if (
            self.train_loader is None
            or self.val_loader is None
            or self.model is None
            or self.optimizer is None
        ):
            raise RuntimeError(
                "Call build_data(fold), build_model(), build_optimizer() before fit()."
            )

        if start_epoch <= 1:
            self.history = []
            self.best_val_pcc = -float("inf")
            self.best_val_loss = float("inf")
        elif start_epoch > 1:
            self.load_checkpoint(fold, start_epoch)
            self.history = self.history[: start_epoch - 1]
            self.best_val_pcc = max(self.history, key=lambda x: x["val_pcc"])["val_pcc"]
            self.best_val_loss = min(self.history, key=lambda x: x["val_mse"])[
                "val_mse"
            ]

        log_step(self.logger, "start training")
        for epoch in range(start_epoch, self.config.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)
            merged = {**train_metrics, **val_metrics}
            self.history.append(merged)

            log_stat(self.logger, f"epoch_{epoch}_train_mse", merged["train_mse"])
            log_stat(self.logger, f"epoch_{epoch}_val_mse", merged["val_mse"])
            log_stat(self.logger, f"epoch_{epoch}_train_pcc", merged["train_pcc"])
            log_stat(self.logger, f"epoch_{epoch}_val_pcc", merged["val_pcc"])

            score = (
                merged["val_pcc"] if np.isfinite(merged["val_pcc"]) else -float("inf")
            )
            if score > self.best_val_pcc:
                self.best_val_pcc = score
                self.best_val_loss = merged["val_mse"]
                self.save_checkpoint(fold=fold, epoch=epoch)

        return self.history

    @torch.no_grad()
    def evaluate(
        self, phase: str = "test", fold: int = None, start_epoch: int = None
    ) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model must be initialized.")

        if phase == "train":
            loader = self.train_loader
        elif phase == "val":
            loader = self.val_loader
        elif phase == "test":
            loader = self.test_loader
        else:
            raise ValueError("phase must be one of: train, val, test")

        if loader is None:
            raise RuntimeError(f"{phase}_loader is not initialized.")

        self.load_checkpoint(fold, start_epoch)

        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        pred_all: List[torch.Tensor] = []
        target_all: List[torch.Tensor] = []

        for batch in tqdm(loader, desc=f"Phase: {phase}"):
            image = batch["image"].to(self.device)
            target = batch["target_gene"].to(self.device)
            pred = self.model(image)
            loss = self.model.loss_fn(pred, target)
            total_loss += float(loss.detach().item())
            n_batches += 1
            pred_all.append(pred.detach().cpu())
            target_all.append(target.detach().cpu())

        mse = total_loss / max(1, n_batches)
        mean_pcc, top20 = self._build_pcc(pred_all, target_all)
        import json

        with open(f"./_pipeline/results/fold{fold}_{phase}_results.json", "w") as f:
            json.dump(
                {
                    "mse": mse,
                    "mean_pcc": mean_pcc,
                    "average_top20": np.mean([score for _, score in top20]),
                    "num_samples": len(loader.dataset),
                    "top20_pcc": top20,
                },
                f,
            )
        return {"mse": mse, "pcc": mean_pcc, "top20_pcc": top20}

    def save_checkpoint(self, fold: int, epoch: int) -> Path:
        if self.model is None:
            raise RuntimeError("Model must be initialized.")

        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"fold{fold}_epoch{epoch}.pt"

        payload = {
            "epoch": epoch,
            "config": self.config.__dict__,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
            "history": self.history,
            "best_val_pcc": self.best_val_pcc,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, fold: int, epoch: int) -> int:
        path = Path(self.config.out_dir) / f"fold{fold}_epoch{epoch}.pt"
        if self.model is None:
            raise RuntimeError("Model must be initialized before load_checkpoint().")

        payload = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(payload["model_state"])

        if self.optimizer is not None and payload.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        else:
            logger.warning("Optimizer state not found in checkpoint.")

        self.history = payload.get("history", [])
        self.best_val_pcc = float(payload.get("best_val_pcc", -float("inf")))
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        return int(payload.get("epoch", 0))


if __name__ == "__main__":
    cfg = PipelineConfig(epochs=1, batch_size=4)
    pipe = Pipeline(cfg)
    pipe.build_data(fold=0)
    pipe.build_model()
    pipe.build_optimizer()
    pipe.fit(fold=0)
    print(pipe.evaluate("test"))
