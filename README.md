# Minimal Reproduction Package for Phononic Crystal Inverse Design

This repository accompanies the manuscript *Inverse Design of Customized Bandgap Characteristics in Phononic Crystals Using Boundary-Enhanced Surrogate Models*. It is a deliberately small, executable reproduction package, not a release of the complete research database.

The package demonstrates four parts of the reported workflow:

1. parameterized complex-hole generation;
2. sector-point extraction, void-area calculation, and unit-cell meshing;
3. training of the boundary-enhanced ConvNeXt surrogate on the supplied demonstration data; and
4. construction of a finite-periodic Abaqus input deck with structural infinite elements from one representative unit-cell mesh.

## Release scope

Included:

- the scripts used for geometry generation, preprocessing, mesh generation, surrogate training, and finite-periodic Abaqus deck construction;
- 24 representative, labelled 128 x 128 unit-cell images in the original training-file layout;
- a tabular manifest containing centre-frequency and inter-bandgap labels for those 24 samples;
- one representative C3D6 unit-cell mesh and a corresponding band-structure table.

Not included:

- the complete 7,608-sample image/label database;
- the complete parameter-to-image mapping and bulk candidate-generation tables;
- trained model checkpoints, the full FEM job archives, ODB files, and batch solver outputs; or
- full active-learning candidate histories and the original propagation video.

The 24 samples are selected only to make file formats and code paths testable. They are insufficient to reproduce the manuscript's reported accuracy statistics or to train a deployable surrogate. Do not compare metrics from this demonstration subset with the manuscript's full-dataset results.

## Layout

```text
data/demo_images/        24 representative 128 x 128 PNG unit cells
data/demo_labels.csv     headerless labels in the original loader-compatible format
data/demo_manifest.csv   human-readable labels and recorded reference predictions
src/                     released research scripts
abaqus/unit_cell/        one representative C3D6 unit-cell input mesh
results/                 representative band-structure output
```

## Environment

Use Python 3.10 or later. Install the packages listed in `requirements.txt`. The published scripts have been tested with PyTorch and torchvision; Abaqus itself is required only to solve the generated `.inp` deck.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Quick checks

Verify the released sample set and its labels:

```bash
python src/verify_release.py
```

Run a short CPU smoke training job. This command verifies the released model code and data interface; it is not intended to reproduce manuscript-level performance.

```bash
python src/train_physics_boundary_convnext.py \
  --csv-path data/demo_labels.csv \
  --image-dir data/demo_images \
  --output-dir outputs/demo \
  --epochs 2 --batch-size 4 --num-workers 0 --device cpu \
  --no-pretrained --no-ema --no-tta-eval
```

Build a representative finite-periodic Abaqus deck (this step only writes files; it does not require Abaqus/CAE):

```bash
python src/build_finite_periodic_abaqus_model.py \
  abaqus/unit_cell/demo_unit_cell_c3d6.inp 8 \
  --freq-min 40000 --freq-max 70000 --freq-points 151 \
  -o outputs/demo_finite_periodic.inp
```

Then submit the resulting deck with an Abaqus installation, for example:

```bash
abaqus job=demo_finite_periodic input=outputs/demo_finite_periodic.inp interactive
```

## Data and code access

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md) for details of the materials provided here. We hope that this package is helpful for readers assessing the workflow, and we welcome reasonable requests for additional materials through the corresponding author.

## Citation

If these materials are useful in your work, please cite the associated manuscript when available.
