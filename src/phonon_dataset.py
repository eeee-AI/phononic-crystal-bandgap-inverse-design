from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


def _parse_float(value: str) -> float:
    text = str(value).strip().replace("Hz", "").replace(",", "")
    return float(text)


def load_phonon_records(
    csv_path: str | Path,
    image_dir: str | Path,
    name_col: int = 0,
    target_col: int = 10,
) -> list[tuple[Path, float, str]]:
    """Read image names and median bandgap frequencies from the CSV.

    The provided CSV has no header: column 1 is the image stem and column 11 is
    the median bandgap frequency. Rows without matching images are skipped.
    """
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    records: list[tuple[Path, float, str]] = []
    missing = 0

    last_decode_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError as exc:
            last_decode_error = exc
    else:
        raise last_decode_error or UnicodeDecodeError("unknown", b"", 0, 1, "cannot decode CSV")

    for row_id, row in enumerate(rows, start=1):
        if len(row) <= max(name_col, target_col):
            continue
        stem = row[name_col].strip()
        if not stem:
            continue
        image_path = image_dir / f"{stem}.png"
        if not image_path.exists():
            missing += 1
            continue
        try:
            target = _parse_float(row[target_col])
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse target at CSV row {row_id}, column {target_col + 1}: {row[target_col]!r}"
            ) from exc
        records.append((image_path, target, stem))

    if not records:
        raise RuntimeError(
            f"No usable samples found. Check csv_path={csv_path} and image_dir={image_dir}."
        )
    if missing:
        print(f"[dataset] skipped {missing} CSV rows because matching PNG files were not found.")
    return records


class PhononHoleDataset(Dataset):
    def __init__(
        self,
        records: list[tuple[Path, float, str]],
        transform: Callable | None = None,
        target_mean: float | None = None,
        target_std: float | None = None,
    ) -> None:
        self.records = records
        self.transform = transform
        targets = torch.tensor([r[1] for r in records], dtype=torch.float32)
        self.target_mean = float(targets.mean()) if target_mean is None else float(target_mean)
        self.target_std = float(targets.std().clamp_min(1e-6)) if target_std is None else float(target_std)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, target, stem = self.records[index]
        image = Image.open(image_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        y = (float(target) - self.target_mean) / self.target_std
        return {
            "image": image,
            "target": torch.tensor([y], dtype=torch.float32),
            "target_hz": torch.tensor([target], dtype=torch.float32),
            "name": stem,
        }
