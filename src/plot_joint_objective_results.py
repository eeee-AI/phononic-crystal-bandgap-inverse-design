from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures_joint_objective"
MODEL_NAME = "physics_boundary_joint"

COLORS = {
    "blue": "#2F6B9A",
    "orange": "#E76F51",
    "teal": "#2A9D8F",
    "gray": "#6C757D",
    "dark": "#263238",
    "light": "#F6F8FA",
}


@dataclass
class Row:
    run_dir: Path
    round_id: int
    rank: int
    target_hz: float
    pred_freq_hz: float | None
    surrogate_target_hz: float | None
    pred_rbg: float | None
    image_name: str
    fem_mid_hz: float | None
    fem_rbg: float | None
    fem_gap_width_hz: float | None
    error_hz: float | None
    status: str


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "Arial"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def target_from_run(run_dir: Path) -> float:
    m = re.search(r"runs_(\d+(?:p\d+)?)Hz", run_dir.name)
    if not m:
        return math.nan
    return float(m.group(1).replace("p", "."))


def read_rows() -> list[Row]:
    rows: list[Row] = []
    run_dirs = sorted(ROOT.glob("runs_*Hz_physics_boundary_joint"), key=target_from_run)
    for run_dir in run_dirs:
        target = target_from_run(run_dir)
        csv_paths = []
        for round_dir in sorted(run_dir.glob("round_*")):
            candidates = sorted(
                round_dir.glob("candidates_feedback*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                csv_paths.append(candidates[0])
        if not csv_paths:
            pool = run_dir / "active_learning_pool.csv"
            if pool.exists():
                csv_paths.append(pool)

        for csv_path in csv_paths:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                for raw in csv.DictReader(f):
                    fem_mid = parse_float(raw.get("fem_mid_hz"))
                    rows.append(
                        Row(
                            run_dir=run_dir,
                            round_id=int(raw.get("round_id", "0") or 0),
                            rank=int(raw.get("rank", "0") or 0),
                            target_hz=target,
                            pred_freq_hz=parse_float(raw.get("pred_freq_hz")),
                            surrogate_target_hz=parse_float(raw.get("surrogate_target_hz")),
                            pred_rbg=parse_float(raw.get("pred_rbg")),
                            image_name=raw.get("image_name", ""),
                            fem_mid_hz=fem_mid,
                            fem_rbg=parse_float(raw.get("fem_rbg")),
                            fem_gap_width_hz=parse_float(raw.get("fem_gap_width_hz")),
                            error_hz=abs(fem_mid - target) if fem_mid is not None and not math.isnan(target) else None,
                            status=raw.get("status", ""),
                        )
                    )
    return rows


def valid_rows(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.error_hz is not None and r.status == "fem_valid"]


def best_by_round(rows: list[Row]) -> list[Row]:
    out = []
    for key in sorted({(r.target_hz, r.round_id) for r in rows}):
        valid = [r for r in rows if (r.target_hz, r.round_id) == key and r.error_hz is not None]
        if valid:
            out.append(min(valid, key=lambda r: r.error_hz or math.inf))
    return out


def best_by_target(rows: list[Row]) -> list[Row]:
    out = []
    for target in sorted({r.target_hz for r in rows}):
        valid = [r for r in rows if r.target_hz == target and r.error_hz is not None]
        if valid:
            out.append(min(valid, key=lambda r: r.error_hz or math.inf))
    return out


def savefig(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_convergence(round_best: list[Row]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for target in sorted({r.target_hz for r in round_best}):
        subset = [r for r in round_best if r.target_hz == target]
        ax.plot(
            [r.round_id for r in subset],
            [r.error_hz for r in subset],
            marker="o",
            linewidth=1.4,
            label=f"{target/1000:.0f} kHz",
        )
    ax.axhline(1000, color=COLORS["gray"], linestyle="--", linewidth=0.9, label="1 kHz")
    ax.axhline(1500, color=COLORS["gray"], linestyle=":", linewidth=0.9, label="1.5 kHz")
    ax.set_xlabel("Round")
    ax.set_ylabel("Best FEM error (Hz)")
    ax.set_title("Joint-objective physics ConvNeXt FEM error convergence")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, ncol=4)
    savefig(fig, "Fig1_physics_boundary_joint_error_convergence")


def plot_target_achievement(best_targets: list[Row]) -> None:
    targets = np.array([r.target_hz for r in best_targets], dtype=float)
    fems = np.array([r.fem_mid_hz or np.nan for r in best_targets], dtype=float)
    errors = np.array([r.error_hz or np.nan for r in best_targets], dtype=float)
    rbgs = np.array([r.fem_rbg or np.nan for r in best_targets], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].scatter(targets / 1000, fems / 1000, s=42, color=COLORS["blue"])
    lo = min(np.nanmin(targets), np.nanmin(fems)) / 1000 - 2
    hi = max(np.nanmax(targets), np.nanmax(fems)) / 1000 + 2
    axes[0].plot([lo, hi], [lo, hi], color=COLORS["dark"], linewidth=1.0)
    axes[0].set_xlabel("Target frequency (kHz)")
    axes[0].set_ylabel("Best FEM median frequency (kHz)")
    axes[0].set_title("Target achievement")
    axes[0].grid(True, color="#E5E7EB", linewidth=0.8)

    x = np.arange(len(best_targets))
    bars = axes[1].bar(x, errors, color=COLORS["orange"], width=0.58, label="error")
    axes[1].set_xticks(x, [f"{r.target_hz/1000:.0f}" for r in best_targets])
    axes[1].set_xlabel("Target frequency (kHz)")
    axes[1].set_ylabel("Minimum FEM error (Hz)")
    axes[1].axhline(1000, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    ax2 = axes[1].twinx()
    ax2.plot(x, rbgs, color=COLORS["teal"], marker="o", linewidth=1.2, label="RBG")
    ax2.set_ylabel("FEM RBG")
    axes[1].set_title("Error and RBG")
    axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    for bar, err in zip(bars, errors):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{err:.0f}", ha="center", va="bottom", fontsize=7)
    savefig(fig, "Fig2_physics_boundary_joint_target_achievement")


def find_image(row: Row) -> Path | None:
    for p in row.run_dir.glob(f"round_{row.round_id:02d}/candidate_images/{row.image_name}"):
        if p.exists():
            return p
    matches = list(row.run_dir.glob(f"round_{row.round_id:02d}/**/{row.image_name}"))
    return matches[0] if matches else None


def plot_best_structures(best_targets: list[Row]) -> None:
    if not best_targets:
        return
    fig, axes = plt.subplots(1, len(best_targets), figsize=(1.55 * len(best_targets), 2.3))
    if len(best_targets) == 1:
        axes = [axes]
    for ax, row in zip(axes, best_targets):
        image_path = find_image(row)
        if image_path and image_path.exists():
            ax.imshow(Image.open(image_path).convert("L"), cmap="gray")
        else:
            ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8, fc=COLORS["light"], ec=COLORS["gray"], lw=0.8))
        ax.set_axis_off()
        ax.set_title(f"{row.target_hz/1000:.0f} kHz\nerr={row.error_hz:.0f} Hz", fontsize=8)
    fig.suptitle("Joint-objective physics ConvNeXt best inverse-designed unit cells", fontsize=10)
    fig.tight_layout()
    savefig(fig, "Fig3_physics_boundary_joint_best_unit_cells")


def write_summary(round_best: list[Row], target_best: list[Row]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "physics_boundary_joint_round_best_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_hz", "round", "rank", "fem_mid_hz", "error_hz", "fem_rbg", "image_name"])
        for r in round_best:
            writer.writerow([r.target_hz, r.round_id, r.rank, r.fem_mid_hz, r.error_hz, r.fem_rbg, r.image_name])
    with (OUT / "physics_boundary_joint_inverse_design_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_hz", "best_round", "rank", "fem_mid_hz", "error_hz", "fem_rbg", "image_name"])
        for r in target_best:
            writer.writerow([r.target_hz, r.round_id, r.rank, r.fem_mid_hz, r.error_hz, r.fem_rbg, r.image_name])


def main() -> None:
    setup_style()
    rows = read_rows()
    if not rows:
        raise SystemExit(f"No Joint-objective physics ConvNeXt run data found under {ROOT}")
    round_best = best_by_round(rows)
    target_best = best_by_target(rows)
    write_summary(round_best, target_best)
    plot_convergence(round_best)
    plot_target_achievement(target_best)
    plot_best_structures(target_best)
    print(f"[done] Joint-objective physics ConvNeXt figures written to {OUT}")


if __name__ == "__main__":
    main()


