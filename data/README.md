# Demonstration data

`demo_labels.csv` has no header by design, matching the source training loader. Column 1 is the image stem and column 11 is the centre-frequency target in Hz. Each stem resolves to `demo_images/<stem>.png`.

`demo_manifest.csv` gives named fields for the same samples. `reference_prediction_hz` and `reference_abs_error_hz` are archived outputs of the full-study model for traceability only; no model checkpoint is distributed in this release.

These 24 records are a representative sample, not a train/test split and not a substitute for the complete 7,608-sample dataset.
