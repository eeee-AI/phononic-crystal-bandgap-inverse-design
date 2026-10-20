# Phononic Crystal Bandgap Inverse Design

This repository accompanies the manuscript *Inverse Design of Customized Bandgap Characteristics in Phononic Crystals Using Boundary-Enhanced Surrogate Models*. It provides code and data files for the reported workflow.

The repository includes:

1. parameterized complex-hole generation;
2. sector-point extraction, void-area calculation, and unit-cell meshing;
3. boundary-enhanced ConvNeXt surrogate training; and
4. finite-periodic Abaqus input-deck construction.

## Contents

- scripts for geometry generation, preprocessing, mesh generation, surrogate training, and finite-periodic Abaqus deck construction;
- 24 labelled 128 x 128 unit-cell images with centre-frequency and inter-bandgap records;
- a representative C3D6 unit-cell mesh and band-structure output; and
- data and code availability notes.

## Layout

```text
data/demo_images/        128 x 128 PNG unit-cell images
data/demo_labels.csv     labels in loader-compatible format
data/demo_manifest.csv   labelled sample manifest
src/                     workflow scripts
abaqus/unit_cell/        C3D6 unit-cell input mesh
results/                 representative band-structure output
```

## Environment

Use Python 3.10 or later and install the packages in `requirements.txt`. PyTorch and torchvision are used for model training. Abaqus is used to solve the generated input deck.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Quick start

Verify the supplied images and labels:

```bash
python src/verify_release.py
```

Run a short training example:

```bash
python src/train_physics_boundary_convnext.py \
  --csv-path data/demo_labels.csv \
  --image-dir data/demo_images \
  --output-dir outputs/demo \
  --epochs 2 --batch-size 4 --num-workers 0 --device cpu \
  --no-pretrained --no-ema --no-tta-eval
```

Build a finite-periodic Abaqus input deck:

```bash
python src/build_finite_periodic_abaqus_model.py \
  abaqus/unit_cell/demo_unit_cell_c3d6.inp 8 \
  --freq-min 40000 --freq-max 70000 --freq-points 151 \
  -o outputs/demo_finite_periodic.inp
```

Run the generated input deck with Abaqus:

```bash
abaqus job=demo_finite_periodic input=outputs/demo_finite_periodic.inp interactive
```

## Joint-objective inverse design

Run the residual-calibrated differential-evolution search for a target centre frequency and relative bandgap:

```bash
python src/active_learning_inverse_design_joint_objective.py \
  --target-freq 50000 --min-rbg 0.015 \
  --physics-boundary-ckpt <physics_boundary_checkpoint.pt> \
  --rbg-model <relative_bandgap_model.pth> \
  --skip-inp
```

Use `--run-abaqus` to generate and evaluate the selected candidates with Abaqus. The round-wise results can be plotted with:

```bash
python src/plot_joint_objective_results.py
```

## Citation

If these materials are useful in your work, please cite the associated manuscript.
