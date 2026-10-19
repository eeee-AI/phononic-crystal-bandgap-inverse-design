"""Check that the minimal public data package is internally consistent."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "demo_images"
LABELS = DATA_DIR / "demo_labels.csv"
MANIFEST = DATA_DIR / "demo_manifest.csv"
EXPECTED_SAMPLE_COUNT = 24


def load_loader_targets() -> dict[str, float]:
    targets: dict[str, float] = {}
    with LABELS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) <= 10:
                raise ValueError("demo_labels.csv must retain the original 11-column layout.")
            targets[row[0].strip()] = float(row[10].replace("Hz", "").strip())
    return targets


def main() -> None:
    targets = load_loader_targets()
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    if len(targets) != EXPECTED_SAMPLE_COUNT or len(manifest) != EXPECTED_SAMPLE_COUNT:
        raise SystemExit(f"Expected {EXPECTED_SAMPLE_COUNT} samples, found {len(targets)} labels and {len(manifest)} manifest rows.")

    errors: list[float] = []
    for item in manifest:
        name = item["sample_id"]
        image = IMAGE_DIR / f"{name}.png"
        if not image.is_file():
            raise SystemExit(f"Missing image: {image}")
        if name not in targets:
            raise SystemExit(f"Manifest sample is absent from demo_labels.csv: {name}")
        if abs(targets[name] - float(item["center_frequency_hz"])) > 1e-6:
            raise SystemExit(f"Centre-frequency mismatch for {name}")
        errors.append(float(item["reference_abs_error_hz"]))

    print(f"Release check passed: {len(manifest)} images and labels are consistent.")
    print(f"Archived full-model MAE on these selected records: {sum(errors) / len(errors):.2f} Hz")
    print("This is a traceability check, not a re-estimated manuscript metric.")


if __name__ == "__main__":
    main()
