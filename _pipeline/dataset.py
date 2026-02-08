from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import get_default_logger, log_stat, log_step


@dataclass
class SectionRecord:
    section_id: str
    count_path: Path
    spot_path: Path
    image_path: Path


class Her2stGenePredictionDataset(Dataset):
    """
    Option 1 (gene reconstruction) dataset.

    Input  : H&E patch centered at spot (112x112 by default).
    Target : 1000-HVG expression vector with preprocessing:
             CPM -> log1p, where CPM = counts / sum(counts_per_spot) * 1e6.
    """

    def __init__(
        self,
        root_dir: str = "data",
        hvg_path: str = "data/her_hvg_cut_1000.npy",
        sections: Optional[List[str]] = None,
        patch_size: int = 112,
        selected_only: bool = True,
        image_transform: Optional[Callable[[Image.Image], object]] = None,
        logger=None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.patch_size = int(patch_size)
        self.selected_only = bool(selected_only)
        self.image_transform = image_transform
        self.logger = logger or get_default_logger()

        log_step(self.logger, "loading HVG list")
        self.hvg_genes = self._load_hvgs(Path(hvg_path))
        log_stat(self.logger, "hvg_total", len(self.hvg_genes))

        log_step(self.logger, "discovering sections")
        self.records = self._discover_sections(sections)
        log_stat(self.logger, "sections_found", len(self.records))

        log_step(self.logger, "building samples")
        self.samples, self.feature_genes = self._build_samples()
        log_stat(self.logger, "samples_total", len(self.samples))
        log_stat(self.logger, "feature_genes_total", len(self.feature_genes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]

        with Image.open(sample["image_path"]) as img:
            img_rgb = img.convert("RGB")
            patch = self._crop_patch(
                img_rgb, sample["pixel_x"], sample["pixel_y"], self.patch_size
            )

        if self.image_transform is not None:
            image_out = self.image_transform(patch)
        else:
            image_np = np.ascontiguousarray(
                np.asarray(patch, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )
            image_out = torch.from_numpy(image_np) if torch is not None else image_np

        gene_np = sample["target"].astype(np.float32)

        target_out = torch.from_numpy(gene_np)
        coord_out = torch.tensor([sample["x"], sample["y"]], dtype=torch.float32)
        pixel_coord_out = torch.tensor(
            [sample["pixel_x"], sample["pixel_y"]], dtype=torch.float32
        )

        return {
            "section_id": sample["section_id"],
            "spot_id": sample["spot_id"],
            "image": image_out,
            "target_gene": target_out,
            "coord": coord_out,
            "pixel_coord": pixel_coord_out,
        }

    def dataset_report(self) -> Dict[str, object]:
        section_counts: Dict[str, int] = {}
        for sample in self.samples:
            sid = sample["section_id"]
            section_counts[sid] = section_counts.get(sid, 0) + 1

        return {
            "root_dir": str(self.root_dir),
            "num_sections": len(section_counts),
            "num_samples": len(self.samples),
            "patch_size": self.patch_size,
            "selected_only": self.selected_only,
            "hvg_requested": len(self.hvg_genes),
            "hvg_present": len(self.feature_genes),
            "missing_hvg": len(self.hvg_genes) - len(self.feature_genes),
            "section_sample_counts": dict(sorted(section_counts.items())),
        }

    def _discover_sections(self, sections: Optional[List[str]]) -> List[SectionRecord]:
        cnt_dir = self.root_dir / "ST-cnts"
        spot_dir = self.root_dir / "ST-spotfiles"
        img_dir = self.root_dir / "ST-imgs"

        requested = set(sections) if sections else None
        records: List[SectionRecord] = []

        for count_path in sorted(cnt_dir.glob("*.tsv")):
            section_id = count_path.stem
            if requested is not None and section_id not in requested:
                continue

            spot_path = spot_dir / f"{section_id}_selection.tsv"
            if not spot_path.exists():
                self.logger.warning("missing spot file for %s", section_id)
                continue

            image_path = self._find_image_path(img_dir, section_id)
            if image_path is None:
                self.logger.warning("missing image for %s", section_id)
                continue

            records.append(
                SectionRecord(
                    section_id=section_id,
                    count_path=count_path,
                    spot_path=spot_path,
                    image_path=image_path,
                )
            )

        return records

    def _build_samples(self):
        all_samples: List[Dict[str, object]] = []
        feature_genes: Optional[List[str]] = None

        from tqdm import tqdm

        for rec in tqdm(self.records, desc="Building Samples"):
            section_samples, section_feature_genes = self._build_section_samples(rec)

            if feature_genes is None:
                feature_genes = section_feature_genes
            elif feature_genes != section_feature_genes:
                raise ValueError(f"HVG feature mismatch in section {rec.section_id}")

            all_samples.extend(section_samples)
            # log_stat(self.logger, f"samples_{rec.section_id}", len(section_samples))

        return all_samples, (feature_genes or [])

    def _build_section_samples(self, rec: SectionRecord):
        cnt = pd.read_csv(rec.count_path, sep="\t", header=0, index_col=0)
        cnt.index = pd.Index([str(x) for x in cnt.index])

        present_hvgs = [g for g in self.hvg_genes if g in cnt.columns]
        if not present_hvgs:
            raise ValueError(
                f"No HVGs found in count matrix for section {rec.section_id}"
            )

        counts = cnt.loc[:, present_hvgs].to_numpy(dtype=np.float32)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cpm = counts / row_sums * 1_000_000.0
        expr = np.log1p(cpm)

        norm_cnt = pd.DataFrame(expr, index=cnt.index, columns=present_hvgs)

        spots = pd.read_csv(rec.spot_path, sep="\t", header=0)
        spots["spot_id"] = spots.apply(lambda r: f"{int(r['x'])}x{int(r['y'])}", axis=1)
        if self.selected_only and "selected" in spots.columns:
            spots = spots.loc[spots["selected"].astype(int) == 1].copy()

        merged = spots.merge(norm_cnt, left_on="spot_id", right_index=True, how="inner")

        samples: List[Dict[str, object]] = []
        for _, row in merged.iterrows():
            samples.append(
                {
                    "section_id": rec.section_id,
                    "spot_id": str(row["spot_id"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "pixel_x": float(row["pixel_x"]),
                    "pixel_y": float(row["pixel_y"]),
                    "target": row[present_hvgs].to_numpy(dtype=np.float32),
                    "image_path": rec.image_path,
                }
            )

        return samples, present_hvgs

    @staticmethod
    def _load_hvgs(hvg_path: Path) -> List[str]:
        if not hvg_path.exists():
            raise FileNotFoundError(f"HVG file not found: {hvg_path}")

        hvgs = np.load(hvg_path, allow_pickle=True)
        return [str(g) for g in hvgs.tolist()]

    @staticmethod
    def _find_image_path(img_dir: Path, section_id: str) -> Optional[Path]:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            matches = list(img_dir.rglob(f"{section_id}/{ext}"))
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _crop_patch(
        image: Image.Image, center_x: float, center_y: float, patch_size: int
    ) -> Image.Image:
        half = patch_size // 2
        cx = int(round(center_x))
        cy = int(round(center_y))

        left = cx - half
        upper = cy - half
        right = left + patch_size
        lower = upper + patch_size

        if left >= 0 and upper >= 0 and right <= image.width and lower <= image.height:
            return image.crop((left, upper, right, lower))

        out = Image.new("RGB", (patch_size, patch_size), color=(0, 0, 0))
        src_left = max(0, left)
        src_upper = max(0, upper)
        src_right = min(image.width, right)
        src_lower = min(image.height, lower)
        cropped = image.crop((src_left, src_upper, src_right, src_lower))

        dst_left = src_left - left
        dst_upper = src_upper - upper
        out.paste(cropped, (dst_left, dst_upper))
        return out


def load_and_report_dataset(
    fold: int,
    phase: str,
    root_dir: str = "./data",
    hvg_path: str = "./data/her_hvg_cut_1000.npy",
    sections: Optional[List[str]] = None,
    patch_size: int = 112,
    selected_only: bool = True,
):

    import yaml

    with open("./_pipeline/cv_splits.yaml", "r") as f:
        cv_splits = yaml.safe_load(f)
    sections = cv_splits["splits"][fold][phase]

    dataset = Her2stGenePredictionDataset(
        root_dir=root_dir,
        hvg_path=hvg_path,
        sections=sections,
        patch_size=patch_size,
        selected_only=selected_only,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    # for batch in loader:
    print(len(loader.dataset))
    return dataset, dataset.dataset_report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--phase", type=str, default="val")
    args = parser.parse_args()

    ds, report = load_and_report_dataset(fold=args.fold, phase=args.phase)
    print(report)
    if len(ds) > 0:
        sample = ds[0]
        print(
            {
                "section_id": sample["section_id"],
                "spot_id": sample["spot_id"],
                "image_shape": tuple(sample["image"].shape),
                "target_shape": tuple(sample["target_gene"].shape),
            }
        )
