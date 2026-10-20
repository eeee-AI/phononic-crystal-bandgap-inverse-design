from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import differential_evolution
from torchvision import transforms
from torchvision.models import convnext_tiny
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR
PHYSICS_MODEL_DIR = SRC_DIR
DEFAULT_ENV3_PYTHON = Path(sys.executable)
sys.path.insert(0, str(SRC_DIR))

from phonon_models import build_model
from train_physics_boundary_convnext import PhysicsBoundaryConvNeXt


class ConvNeXtRegression(nn.Module):
    def __init__(self, num_outputs: int = 1):
        super().__init__()
        self.model = convnext_tiny(weights=None)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_outputs),
        )

    def forward(self, x):
        return self.model(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.channel_attn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial_attn(torch.cat([avg_out, max_out], dim=1))


def attach_cbam_to_resnet18(model):
    import types

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for i in range(len(layer)):
            out_channels = layer[i].bn2.num_features
            layer[i].cbam = CBAM(out_channels)

            def new_forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.cbam(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
                out = self.relu(out)
                return out

            layer[i].forward = types.MethodType(new_forward, layer[i])
    return model


@dataclass
class Candidate:
    round_id: int
    rank: int
    params: list[float]
    pred_freq_hz: float
    pred_freq_std_hz: float
    pred_freq_members: str
    surrogate_target_hz: float
    residual_correction_hz: float
    pred_rbg: float
    pred_gap_width_hz: float
    surrogate_score: float
    image_name: str
    sector_name: str
    prefix: str
    fem_mid_hz: float | None = None
    fem_rbg: float | None = None
    fem_gap_width_hz: float | None = None
    fem_error_hz: float | None = None
    status: str = "surrogate_selected"


@dataclass
class FrequencyPredictor:
    name: str
    model: nn.Module
    target_mean: float
    target_std: float


class ActiveLearningInverseDesigner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.work_dir = self.resolve_work_dir(args)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        print(f"[output] work_dir={self.work_dir}")
        self.warn_if_work_dir_target_mismatch()
        self.freq_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.rbg_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.freq_predictors = self._load_frequency_ensemble()
        self.rbg_model = self._load_convnext_model(Path(args.rbg_model), "RBG")
        self.geometry_filter = self._load_geometry_filter(Path(args.geometry_filter)) if args.geometry_filter else None
        self.history: list[Candidate] = []
        self.base_bounds = [(5.0, 15.0), (5.0, 15.0), (2.0, 10.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0)]
        self.current_bounds = list(self.base_bounds)
        self.residual_correction_hz = 0.0

    @classmethod
    def resolve_work_dir(cls, args: argparse.Namespace) -> Path:
        if args.work_dir is not None:
            return Path(args.work_dir)
        tag = cls.target_freq_tag(args.target_freq)
        suffix = "physics_boundary_joint" if args.freq_aggregation == "member" else f"{args.freq_aggregation}_joint"
        output_root = Path(getattr(args, "output_root", EXPERIMENT_DIR) or EXPERIMENT_DIR)
        return output_root / f"runs_{tag}_{suffix}"

    @staticmethod
    def target_freq_tag(freq_hz: float) -> str:
        if abs(freq_hz - round(freq_hz)) < 1e-6:
            text = str(int(round(freq_hz)))
        else:
            text = f"{freq_hz:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"{text}Hz"

    def warn_if_work_dir_target_mismatch(self) -> None:
        expected = self.target_freq_tag(float(self.args.target_freq))
        if expected not in self.work_dir.name:
            print(
                f"[warning] work_dir name does not contain target tag {expected}: {self.work_dir.name}. "
                "This is allowed, but it can mix results from different target frequencies."
            )

    def _load_convnext_model(self, path: Path, label: str) -> nn.Module:
        model = ConvNeXtRegression().to(self.device)
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        print(f"[model] loaded {label} model: {path}")
        return model

    def _load_frequency_ensemble(self) -> list[FrequencyPredictor]:
        predictors = [self._load_physics_boundary_checkpoint(Path(self.args.physics_boundary_ckpt))]
        print("[model] frequency ensemble:", ", ".join(p.name for p in predictors))
        return predictors

    def _load_physics_boundary_checkpoint(self, path: Path) -> FrequencyPredictor:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for physics_boundary_convnext: {path}")
        ckpt = torch.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise ValueError(f"{path} is not a compatible physics_boundary_convnext checkpoint.")
        saved_args = ckpt.get("args", {})
        model = PhysicsBoundaryConvNeXt(
            pretrained=bool(saved_args.get("pretrained", True)),
            dropout=float(saved_args.get("dropout", 0.12)),
        ).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        target_mean = float(ckpt["target_mean"])
        target_std = float(ckpt["target_std"])
        print(f"[model] loaded frequency member physics_boundary_convnext: {path}")
        return FrequencyPredictor("physics_boundary_convnext", model, target_mean, target_std)

    def _load_frequency_checkpoint(self, model_name: str, path: Path) -> FrequencyPredictor:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {path}")
        ckpt = torch.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise ValueError(f"{path} is not a phonon_train.py checkpoint. Please retrain it with phonon_train.py.")
        saved_model_name = ckpt.get("args", {}).get("model", model_name)
        if saved_model_name != model_name:
            print(f"[model] warning: requested {model_name}, checkpoint says {saved_model_name}; using checkpoint model name.")
            model_name = saved_model_name
        model = build_model(model_name, pretrained=False).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        target_mean = float(ckpt["target_mean"])
        target_std = float(ckpt["target_std"])
        print(f"[model] loaded frequency member {model_name}: {path}")
        return FrequencyPredictor(model_name, model, target_mean, target_std)

    def _load_frequency_model(self, path: Path) -> tuple[nn.Module, float | None, float | None]:
        if self.args.freq_model_kind == "holebandnet":
            ckpt = torch.load(path, map_location=self.device)
            model_name = ckpt.get("args", {}).get("model", "holebandnet")
            model = build_model(model_name, pretrained=False).to(self.device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            target_mean = float(ckpt["target_mean"])
            target_std = float(ckpt["target_std"])
            print(f"[model] loaded frequency model ({model_name}): {path}")
            print(f"[model] frequency target_mean={target_mean:.6f}, target_std={target_std:.6f}")
            return model, target_mean, target_std
        if self.args.freq_model_kind == "convnext":
            return self._load_convnext_model(path, "frequency"), None, None
        raise ValueError("--freq-model-kind must be 'holebandnet' or 'convnext'")

    def _load_geometry_filter(self, path: Path) -> nn.Module:
        model = models.resnet18(weights=None)
        model = attach_cbam_to_resnet18(model)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"[model] loaded geometry legality filter: {path}")
        return model

    @staticmethod
    def get_full_pts(params):
        r0, r1, r2 = params[0], params[1], params[2]
        p0 = np.array([r0, 0])
        p1 = np.array([r1 * np.cos(np.pi / 8), r1 * np.sin(np.pi / 8)])
        p2 = np.array([r2 * np.cos(np.pi / 4), r2 * np.sin(np.pi / 4)])
        p3 = np.array([p1[1], p1[0]])
        p4 = np.array([0, r0])
        pts = [
            p0, p1, p2, p3, p4, [-p3[0], p3[1]], [-p2[0], p2[1]], [-p1[0], p1[1]], [-p0[0], p0[1]],
            [-p1[0], -p1[1]], [-p2[0], -p2[1]], [-p3[0], -p3[1]], [-p4[0], -p4[1]],
            [p3[0], -p3[1]], [p2[0], -p2[1]], [p1[0], -p1[1]],
        ]
        return [np.array(pt, dtype=float) for pt in pts]

    @staticmethod
    def cubic_bezier(p0, p1, p2, p3, num_points=30):
        t = np.linspace(0, 1, num_points)[:, np.newaxis]
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

    @classmethod
    def create_closed_bezier_curve(cls, control_points, base_tensions):
        n = len(control_points)
        curve_points = []
        for i in range(n):
            p0, p3 = control_points[i], control_points[(i + 1) % n]
            p_prev, p_next = control_points[(i - 1) % n], control_points[(i + 2) % n]

            def norm_v(v):
                norm = np.linalg.norm(v)
                return v / norm if norm > 1e-6 else v

            t1 = norm_v(norm_v(p0 - p_prev) + norm_v(p3 - p0))
            t2 = norm_v(norm_v(p3 - p0) + norm_v(p_next - p3))
            angle = np.degrees(np.arctan2((p0 + p3)[1] / 2, (p0 + p3)[0] / 2)) % 90
            tension = base_tensions[1] if 22.5 <= angle <= 67.5 else base_tensions[0]
            dist = np.linalg.norm(p3 - p0)
            curve_points.extend(cls.cubic_bezier(p0, p0 + tension * dist * t1, p3 - tension * dist * t2, p3))
        return np.array(curve_points)

    @staticmethod
    def scale_curve_to_envelope(curve, target_side=30):
        max_coord = np.max(np.abs(curve))
        if max_coord <= 1e-9:
            return curve
        return curve * ((target_side / 2) / max_coord)

    @staticmethod
    def polygon_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    def params_to_curve(self, params, scaled: bool = False):
        ctrl_pts = self.get_full_pts(params[:3])
        curve = self.create_closed_bezier_curve(np.array(ctrl_pts), params[3:])
        if scaled:
            curve = self.scale_curve_to_envelope(curve, target_side=30)
        return curve

    def params_to_image(self, params) -> Image.Image | None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=32)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(0, 33.0)
        ax.set_ylim(0, 33.0)
        ax.set_facecolor("white")
        curve = self.params_to_curve(params, scaled=False)
        phys_pts = curve + 16.5
        if np.any(phys_pts < 0.5) or np.any(phys_pts > 32.5):
            plt.close(fig)
            return None
        ax.plot(phys_pts[:, 0], phys_pts[:, 1], color="blue", linewidth=1)
        ax.axis("off")
        fig.canvas.draw()
        img = Image.frombuffer("RGBA", (128, 128), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
        plt.close(fig)
        return img

    @staticmethod
    def has_segment_intersections(points: np.ndarray) -> bool:
        def orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def intersect(a, b, c, d):
            eps = 1e-9
            if max(a[0], b[0]) + eps < min(c[0], d[0]) or max(c[0], d[0]) + eps < min(a[0], b[0]):
                return False
            if max(a[1], b[1]) + eps < min(c[1], d[1]) or max(c[1], d[1]) + eps < min(a[1], b[1]):
                return False
            o1 = orient(a, b, c)
            o2 = orient(a, b, d)
            o3 = orient(c, d, a)
            o4 = orient(c, d, b)
            return o1 * o2 < -eps and o3 * o4 < -eps

        n = len(points)
        if n < 4:
            return True
        for i in range(n):
            a, b = points[i], points[(i + 1) % n]
            for j in range(i + 1, n):
                if abs(i - j) <= 1 or {i, j} == {0, n - 1}:
                    continue
                c, d = points[j], points[(j + 1) % n]
                if intersect(a, b, c, d):
                    return True
        return False

    @torch.no_grad()
    def is_geometry_legal(self, params, img: Image.Image | None = None) -> bool:
        curve = self.params_to_curve(params, scaled=False)
        phys_pts = curve + 16.5
        if np.any(phys_pts < self.args.min_boundary_margin) or np.any(phys_pts > 33.0 - self.args.min_boundary_margin):
            return False
        if self.has_segment_intersections(phys_pts):
            return False
        if self.geometry_filter is not None:
            if img is None:
                img = self.params_to_image(params)
            if img is None:
                return False
            x = self.rbg_transform(img).unsqueeze(0).to(self.device)
            pred = torch.softmax(self.geometry_filter(x), dim=1)[0]
            legal_prob = float(pred[1].item())
            return legal_prob >= self.args.geometry_legal_threshold
        return True

    @torch.no_grad()
    def predict(self, img: Image.Image) -> tuple[float, float, float, dict[str, float]]:
        x_freq = self.freq_transform(img).unsqueeze(0).to(self.device)
        x_rbg = self.rbg_transform(img).unsqueeze(0).to(self.device)
        member_preds = {}
        for predictor in self.freq_predictors:
            pred_raw = float(predictor.model(x_freq).item())
            member_preds[predictor.name] = pred_raw * predictor.target_std + predictor.target_mean
        pred_values = np.array(list(member_preds.values()), dtype=float)
        if self.args.freq_aggregation == "member":
            if self.args.freq_member not in member_preds:
                raise ValueError(f"Unknown --freq-member {self.args.freq_member!r}; available: {sorted(member_preds)}")
            pred_freq = float(member_preds[self.args.freq_member])
        elif self.args.freq_aggregation == "median":
            pred_freq = float(np.median(pred_values))
        else:
            pred_freq = float(pred_values.mean())
        pred_freq_std = float(pred_values.std(ddof=0))
        pred_rbg = float(self.rbg_model(x_rbg).item() * self.args.rbg_scale)
        return pred_freq, pred_freq_std, pred_rbg, member_preds

    def current_surrogate_target(self) -> float:
        return float(self.args.target_freq + self.residual_correction_hz)

    def minimum_gap_width_hz(self, surrogate_target: float) -> float:
        if self.args.target_gap_width_hz > 0:
            return float(self.args.target_gap_width_hz)
        return float(max(0.0, self.args.min_rbg) * max(1.0, surrogate_target))

    def joint_objective_terms(self, pred_freq: float, pred_freq_std: float, pred_rbg: float) -> dict[str, float]:
        surrogate_target = self.current_surrogate_target()
        pred_gap_width = max(0.0, pred_rbg) * max(1.0, pred_freq)
        min_gap_width = self.minimum_gap_width_hz(surrogate_target)
        freq_error = abs(pred_freq - surrogate_target)
        disagreement_penalty = pred_freq_std * self.args.ensemble_std_weight
        if self.args.freq_aggregation == "member" and not self.args.penalize_member_disagreement:
            disagreement_penalty = 0.0
        rbg_penalty = max(0.0, self.args.min_rbg - pred_rbg) * self.args.rbg_penalty
        width_shortfall = max(0.0, min_gap_width - pred_gap_width)
        width_shortfall_penalty = width_shortfall * self.args.gap_width_shortfall_weight
        useful_extra_width = min(max(0.0, pred_gap_width - min_gap_width), self.args.gap_width_reward_cap_hz)
        width_reward = useful_extra_width * self.args.gap_width_reward_weight
        score = (
            freq_error * self.args.freq_error_weight
            + disagreement_penalty
            + rbg_penalty
            + width_shortfall_penalty
            - width_reward
        )
        return {
            "surrogate_target": surrogate_target,
            "pred_gap_width": pred_gap_width,
            "min_gap_width": min_gap_width,
            "freq_error": freq_error,
            "disagreement_penalty": disagreement_penalty,
            "rbg_penalty": rbg_penalty,
            "width_shortfall_penalty": width_shortfall_penalty,
            "width_reward": width_reward,
            "score": score,
        }

    def objective(self, x):
        img = self.params_to_image(x)
        if img is None:
            return 1e9
        if not self.is_geometry_legal(x, img):
            return 5e8
        pred_freq, pred_freq_std, pred_rbg, member_preds = self.predict(img)
        terms = self.joint_objective_terms(pred_freq, pred_freq_std, pred_rbg)
        outside_bonus = 0.0
        if self.args.prefer_uncertain:
            outside_bonus = -self.parameter_distance_bonus(x)
        score = terms["score"] + outside_bonus
        self._candidate_trace.append(
            (np.asarray(x, dtype=float), pred_freq, pred_freq_std, member_preds, pred_rbg, terms["pred_gap_width"], score)
        )
        if len(self._candidate_trace) % 20 == 0:
            print(
                f"  eval {len(self._candidate_trace):04d}: ensemble={pred_freq:.2f}卤{pred_freq_std:.2f} Hz, "
                f"surrogate_target={terms['surrogate_target']:.2f} Hz, "
                f"gap={terms['pred_gap_width']:.1f} Hz, pred_rbg={pred_rbg:.4f}, "
                f"width_reward={terms['width_reward']:.1f}, score={score:.2f}"
            )
        return score

    def parameter_distance_bonus(self, x) -> float:
        center = np.array([10.0, 10.0, 6.0, 0.55, 0.55, 0.55])
        span = np.array([10.0, 10.0, 8.0, 0.9, 0.9, 0.9])
        return float(np.linalg.norm((np.asarray(x) - center) / span) * self.args.uncertain_bonus)

    def run_de(self, round_id: int) -> list[Candidate]:
        self._candidate_trace = []
        print(f"\n[round {round_id}] DE joint-objective search target={self.args.target_freq:.2f} Hz min_rbg={self.args.min_rbg:.4f}")
        print(
            f"[round {round_id}] surrogate target={self.current_surrogate_target():.2f} Hz "
            f"(residual correction={self.residual_correction_hz:.2f} Hz)"
        )
        print(
            f"[round {round_id}] objective=frequency_error*{self.args.freq_error_weight:.2f} "
            f"+ gap_shortfall*{self.args.gap_width_shortfall_weight:.2f} "
            f"- extra_gap*{self.args.gap_width_reward_weight:.2f} "
            f"(extra cap={self.args.gap_width_reward_cap_hz:.0f} Hz)"
        )
        print(f"[round {round_id}] bounds={self.current_bounds}")
        differential_evolution(
            self.objective,
            self.current_bounds,
            strategy="best1bin",
            popsize=self.args.popsize,
            maxiter=self.args.maxiter,
            tol=self.args.de_tol,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            disp=True,
            seed=self.args.seed + round_id,
        )
        unique = {}
        for params, pred_freq, pred_freq_std, member_preds, pred_rbg, pred_gap_width, score in self._candidate_trace:
            key = tuple(np.round(params, 4))
            if key not in unique or score < unique[key][6]:
                unique[key] = (params, pred_freq, pred_freq_std, member_preds, pred_rbg, pred_gap_width, score)
        candidates = []
        ranked = sorted(unique.values(), key=lambda item: item[6])[: self.args.top_k]
        for rank, (params, pred_freq, pred_freq_std, member_preds, pred_rbg, pred_gap_width, score) in enumerate(ranked, start=1):
            prefix = f"al_r{round_id:02d}_{rank:03d}"
            area_percent, image_name, sector_name = self.save_geometry(round_id, rank, params, prefix)
            candidates.append(
                Candidate(
                    round_id=round_id,
                    rank=rank,
                    params=[float(v) for v in params],
                    pred_freq_hz=float(pred_freq),
                    pred_freq_std_hz=float(pred_freq_std),
                    pred_freq_members=json.dumps(member_preds, ensure_ascii=False),
                    surrogate_target_hz=self.current_surrogate_target(),
                    residual_correction_hz=float(self.residual_correction_hz),
                    pred_rbg=float(pred_rbg),
                    pred_gap_width_hz=float(pred_gap_width),
                    surrogate_score=float(score),
                    image_name=image_name,
                    sector_name=sector_name,
                    prefix=f"model_{area_percent}_{prefix}_c3d6_",
                )
            )
        return candidates

    def save_geometry(self, round_id: int, rank: int, params, prefix: str) -> tuple[int, str, str]:
        round_dir = self.work_dir / f"round_{round_id:02d}"
        curve_dir = round_dir / "full_curve_new"
        curve_dir.mkdir(parents=True, exist_ok=True)
        curve = self.params_to_curve(params, scaled=True)
        area_percent = int(self.polygon_area(curve) / (33 * 33) * 100)
        image_name = f"{area_percent}_{prefix}.png"
        sector_name = f"{area_percent}_{prefix}_sector.txt"
        self.plot_curve(curve, curve_dir / image_name)
        sector = self.extract_sector_points(curve)
        with (curve_dir / sector_name).open("w", encoding="utf-8") as f:
            f.write("x,y\n")
            for x, y in sector:
                f.write(f"{x:.6f},{y:.6f}\n")
        return area_percent, image_name, sector_name

    @staticmethod
    def extract_sector_points(curve):
        pts = [[16.5, 0.0]]
        for x, y in curve:
            if x >= -1e-8 and y >= -1e-8 and y <= x + 1e-8:
                pts.append([float(x), float(y)])
        pts.append([16.5, 16.5])
        return np.array(pts)

    @staticmethod
    def plot_curve(curve, save_path: Path):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.add_patch(plt.Rectangle((-16.5, -16.5), 33, 33, fill=False, color="black", linewidth=1))
        ax.fill(curve[:, 0], curve[:, 1], color="skyblue", alpha=0.42)
        ax.plot(curve[:, 0], curve[:, 1], color="blue", linewidth=1.3)
        ax.set_xlim(-17, 17)
        ax.set_ylim(-17, 17)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def generate_inp(self, round_id: int) -> int:
        round_dir = self.work_dir / f"round_{round_id:02d}"
        curve_dir = round_dir / "full_curve_new"
        print(f"[round {round_id}] generating inp from sector files...")
        sector_files = sorted(curve_dir.glob("*_sector.txt"))
        inp_dir = curve_dir / "inp_output"
        inp_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        for sector_file in sector_files:
            print(f"[gmsh] process {sector_file.name}")
            ok = self.run_mesh_subprocess(round_dir, sector_file, inp_dir)
            if not ok:
                print(f"[gmsh] skipped bad candidate: {sector_file.name}")
            else:
                success_count += 1
        prefix_count = self.prepare_abaqus_runner(round_dir)
        if prefix_count == 0:
            print(
                "[gmsh] no Abaqus inp files were generated. "
                "Abaqus will be skipped for this round; check *_mesh.log in inp_output."
            )
        else:
            print(f"[gmsh] generated {prefix_count} Abaqus model groups from {success_count}/{len(sector_files)} candidates.")
        return prefix_count

    def run_mesh_subprocess(self, round_dir: Path, sector_file: Path, inp_dir: Path) -> bool:
        mesh_python = Path(self.args.mesh_python)
        if not mesh_python.exists():
            mesh_python = Path(sys.executable)
        prefix = "model_" + sector_file.stem.replace("_sector", "") + "_c3d6_"
        before = set(self.kpoint_inp_files(inp_dir, prefix))
        log_path = inp_dir / f"{sector_file.stem}_mesh.log"
        code = (
            "import sys, os; "
            f"sys.path.insert(0, {str(DATA_DIR)!r}); "
            "import generate_unit_cell_mesh; "
            f"os.chdir({str(round_dir)!r}); "
            f"generate_unit_cell_mesh.process_sector_file({str(sector_file)!r}, output_dir={str(inp_dir)!r})"
        )
        proc = subprocess.Popen(
            [str(mesh_python), "-c", code],
            cwd=str(round_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            out, _ = proc.communicate(timeout=self.args.mesh_timeout)
            if out:
                self.safe_print(out[-4000:])
                log_path.write_text(out, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            if out:
                self.safe_print(out[-2000:])
                log_path.write_text(out, encoding="utf-8", errors="replace")
            print(f"[gmsh] timeout after {self.args.mesh_timeout}s")
            return False
        after = set(self.kpoint_inp_files(inp_dir, prefix))
        new_files = sorted(after - before)
        if proc.returncode != 0:
            print(f"[gmsh] process failed with code {proc.returncode}; log={log_path}")
            return False
        if not after:
            print(f"[gmsh] process returned ok but no inp matched {prefix}*.inp; log={log_path}")
            return False
        if not new_files:
            print(f"[gmsh] inp files already existed for {prefix}; reusing them.")
        return True

    @staticmethod
    def safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("gbk", errors="replace").decode("gbk"))

    @staticmethod
    def kpoint_inp_files(inp_dir: Path, prefix: str | None = None) -> list[Path]:
        files = inp_dir.glob((prefix or "model_*_c3d6_") + "*.inp")
        return [p for p in files if re.search(r"_c3d6_\d+\.inp$", p.name)]

    def prepare_abaqus_runner(self, round_dir: Path) -> int:
        inp_dir = round_dir / "full_curve_new" / "inp_output"
        prefixes = sorted({p.name.rsplit("_", 1)[0] + "_" for p in self.kpoint_inp_files(inp_dir)})
        prefix_path = inp_dir / "all_prefix.txt"
        prefix_path.write_text("\n".join(prefixes) + "\n", encoding="utf-8")
        for exe_name in ["getdata.exe", "bandgap_flag.exe"]:
            src = DATA_DIR / "full_curve_new" / "inp_output" / exe_name
            if src.exists():
                shutil.copy2(src, inp_dir / exe_name)
        runner = inp_dir / "run_abaqus_batch.py"
        runner.write_text(self.batch_runner_code(prefix_path.name, self.args.abaqus_cmd, self.args.cpus), encoding="utf-8")
        print(f"[abaqus] prepared {runner} with {len(prefixes)} model groups")
        return len(prefixes)

    @staticmethod
    def batch_runner_code(prefix_file: str, abaqus_cmd: str, cpus: int) -> str:
        return f'''import os
import subprocess
import time

prefix_file = r"{prefix_file}"
work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)
to_delete_exts = ['.odb', '.sim', '.res', '.mdl', '.stt', '.com', '.prt', '.log', '.sta', '.msg', '.sel', '.exception', '.simlog']

with open(prefix_file, 'r', encoding='utf-8') as f:
    prefixes = [line.strip() for line in f if line.strip()]

for prefix in prefixes:
    print(f"\\n[{{time.strftime('%H:%M:%S')}}] group: {{prefix}}")
    for i in range(1, 22):
        inp_name = f"{{prefix}}{{i}}.inp"
        job_name = f"job_{{prefix}}{{i}}"
        if not os.path.exists(inp_name):
            continue
        cmd = r"{abaqus_cmd}" + f" job={{job_name}} input={{inp_name}} cpus={cpus} ask_delete=OFF"
        print("  run", cmd)
        subprocess.run(cmd, shell=True)
    time.sleep(3)
    for i in range(1, 22):
        job_name = f"job_{{prefix}}{{i}}"
        for ext in to_delete_exts:
            path = f"{{job_name}}{{ext}}"
            if os.path.exists(path):
                for _ in range(5):
                    try:
                        os.remove(path)
                        break
                    except OSError:
                        time.sleep(2)

for exe in ["getdata.exe", "bandgap_flag.exe"]:
    if os.path.exists(exe):
        print("postprocess", exe)
        subprocess.run(exe, shell=True)
'''

    def run_abaqus(self, round_id: int):
        runner = self.work_dir / f"round_{round_id:02d}" / "full_curve_new" / "inp_output" / "run_abaqus_batch.py"
        prefix_file = runner.parent / "all_prefix.txt"
        if not prefix_file.exists() or not prefix_file.read_text(encoding="utf-8").strip():
            print(f"[abaqus] skipped because {prefix_file} is empty.")
            return
        print(f"[abaqus] running {runner}")
        subprocess.run([sys.executable, str(runner)], check=False)

    def parse_feedback(self, round_id: int, candidates: list[Candidate]) -> None:
        inp_dir = self.work_dir / f"round_{round_id:02d}" / "full_curve_new" / "inp_output"
        bandgap_path = inp_dir / "bandgap.csv"
        if not bandgap_path.exists():
            print(f"[feedback] {bandgap_path} not found. FEM feedback skipped.")
            for c in candidates:
                c.status = "awaiting_fem"
            return
        rows = self.read_bandgap_rows(bandgap_path)
        for c in candidates:
            keys = self.feedback_name_candidates(c)
            inp_dir = self.work_dir / f"round_{round_id:02d}" / "full_curve_new" / "inp_output"
            if not any(inp_dir.glob(c.prefix + "*.inp")):
                c.status = "mesh_failed_or_skipped"
                continue
            matches = [r for r in rows if r["name"] in keys]
            if not matches:
                c.status = "fem_no_bandgap"
                continue
            best = min(
                matches,
                key=lambda r: abs(r["mid_hz"] - self.args.target_freq) + max(0.0, self.args.min_rbg - r["rbg"]) * self.args.rbg_penalty,
            )
            c.fem_mid_hz = best["mid_hz"]
            c.fem_rbg = best["rbg"]
            c.fem_gap_width_hz = best["width_hz"]
            c.fem_error_hz = abs(best["mid_hz"] - self.args.target_freq)
            c.status = "fem_valid" if best["rbg"] >= self.args.min_rbg else "fem_rbg_too_small"

    @staticmethod
    def feedback_name_candidates(candidate: Candidate) -> set[str]:
        """Return possible FEM postprocess names for one candidate.

        The Abaqus jobs are run as job_model_<area>_<id>_c3d6_<kpoint>, but the
        bandgap postprocessor collapses that to job_<area>_<id>. Keep several
        variants so old result folders remain readable.
        """
        prefix = candidate.prefix.rstrip("_")
        keys = {
            "job_" + prefix,
            "job_" + prefix.rstrip("_"),
        }
        match = re.match(r"model_(.+?)_c3d6$", prefix)
        if match:
            base = match.group(1)
            keys.update(
                {
                    "job_" + base,
                    base,
                    "job_model_" + base + "_c3d6",
                }
            )
        image_stem = Path(candidate.image_name).stem
        keys.update({"job_" + image_stem, image_stem})
        return keys

    @staticmethod
    def read_bandgap_rows(path: Path):
        rows = []
        for encoding in ["utf-8-sig", "gb18030", "gbk"]:
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = path.read_text(errors="ignore")
        reader = csv.reader(text.splitlines())
        for row in reader:
            if len(row) < 9:
                continue
            try:
                lower_max = float(row[3].replace("Hz", "").strip())
                upper_min = float(row[5].replace("Hz", "").strip())
                width = float(row[8].replace("Hz", "").strip())
            except ValueError:
                continue
            mid = (lower_max + upper_min) / 2.0
            rows.append({"name": row[0].strip(), "mid_hz": mid, "width_hz": width, "rbg": width / max(mid, 1e-9)})
        return rows

    def write_round_outputs(self, round_id: int, candidates: list[Candidate]):
        round_dir = self.work_dir / f"round_{round_id:02d}"
        json_path = round_dir / "candidates_feedback.json"
        json_path.write_text(json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2), encoding="utf-8")
        csv_path = round_dir / "candidates_feedback.csv"
        self.write_candidates_csv(csv_path, candidates)
        self.append_learning_pool(candidates)
        self.append_convergence_log(round_id, candidates)

    @staticmethod
    def write_candidates_csv(csv_path: Path, candidates: list[Candidate]):
        fieldnames = list(asdict(candidates[0]).keys()) if candidates else []
        try:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for c in candidates:
                    writer.writerow(asdict(c))
        except PermissionError:
            fallback = csv_path.with_name(f"{csv_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{csv_path.suffix}")
            with fallback.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for c in candidates:
                    writer.writerow(asdict(c))
            print(f"[warning] {csv_path} is locked. Wrote fallback CSV: {fallback}")

    def append_learning_pool(self, candidates: list[Candidate]):
        pool = self.work_dir / "active_learning_pool.csv"
        exists = pool.exists()
        fields = list(asdict(candidates[0]).keys()) if candidates else []
        with pool.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for c in candidates:
                writer.writerow(asdict(c))

    def fem_joint_score(self, candidate: Candidate) -> float:
        if candidate.fem_mid_hz is None:
            return float("inf")
        fem_mid = float(candidate.fem_mid_hz)
        fem_rbg = float(candidate.fem_rbg or 0.0)
        if candidate.fem_gap_width_hz is not None:
            gap_width = float(candidate.fem_gap_width_hz)
        else:
            gap_width = fem_rbg * max(1.0, fem_mid)
        min_gap_width = self.minimum_gap_width_hz(self.args.target_freq)
        freq_error = abs(fem_mid - self.args.target_freq)
        rbg_penalty = max(0.0, self.args.min_rbg - fem_rbg) * self.args.rbg_penalty
        width_shortfall_penalty = max(0.0, min_gap_width - gap_width) * self.args.gap_width_shortfall_weight
        useful_extra_width = min(max(0.0, gap_width - min_gap_width), self.args.gap_width_reward_cap_hz)
        width_reward = useful_extra_width * self.args.gap_width_reward_weight
        return float(freq_error * self.args.freq_error_weight + rbg_penalty + width_shortfall_penalty - width_reward)

    def append_convergence_log(self, round_id: int, candidates: list[Candidate]):
        path = self.work_dir / "fem_error_convergence.csv"
        valid = [c for c in candidates if c.fem_error_hz is not None]
        if valid:
            best = min(valid, key=self.fem_joint_score)
            row = {
                "round": round_id,
                "best_rank": best.rank,
                "best_fem_joint_score": self.fem_joint_score(best),
                "best_fem_error_hz": best.fem_error_hz,
                "best_fem_mid_hz": best.fem_mid_hz,
                "best_fem_rbg": best.fem_rbg,
                "best_fem_gap_width_hz": best.fem_gap_width_hz,
                "residual_correction_hz": self.residual_correction_hz,
                "surrogate_target_hz": self.current_surrogate_target(),
                "status": best.status,
            }
            print(
                f"Round {round_id}: best FEM joint score = {self.fem_joint_score(best):.2f} "
                f"(error={best.fem_error_hz:.2f} Hz, mid={best.fem_mid_hz:.2f} Hz, "
                f"gap={best.fem_gap_width_hz}, RBG={best.fem_rbg:.4f}, rank=#{best.rank})"
            )
        else:
            row = {
                "round": round_id,
                "best_rank": "",
                "best_fem_joint_score": "",
                "best_fem_error_hz": "",
                "best_fem_mid_hz": "",
                "best_fem_rbg": "",
                "best_fem_gap_width_hz": "",
                "residual_correction_hz": self.residual_correction_hz,
                "surrogate_target_hz": self.current_surrogate_target(),
                "status": "no_fem_feedback",
            }
            print(f"Round {round_id}: no FEM feedback available yet.")
        exists = path.exists()
        with path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def update_residual_calibration(self, candidates: list[Candidate], round_id: int):
        if not self.args.residual_correction:
            return
        valid = [c for c in candidates if c.fem_mid_hz is not None and c.status == "fem_valid"]
        if not valid:
            print("[calibration] no valid FEM feedback; residual correction unchanged.")
            return

        best = min(
            valid,
            key=self.fem_joint_score,
        )
        old = float(self.residual_correction_hz)
        fem_error_signed = float(self.args.target_freq - float(best.fem_mid_hz))
        step = fem_error_signed * self.args.residual_step_gain
        step = float(np.clip(step, -self.args.max_residual_step_hz, self.args.max_residual_step_hz))
        updated = old + step
        updated = float(np.clip(updated, -self.args.max_residual_correction_hz, self.args.max_residual_correction_hz))
        self.residual_correction_hz = updated
        print(
            f"[calibration] round {round_id}: best FEM mid={best.fem_mid_hz:.2f} Hz, "
            f"signed error={fem_error_signed:.2f} Hz, step={step:.2f} Hz, "
            f"correction {old:.2f} -> {updated:.2f} Hz; next surrogate target={self.current_surrogate_target():.2f} Hz"
        )

    def update_bounds_from_feedback(self, candidates: list[Candidate], round_id: int):
        if not self.args.adaptive_bounds:
            return
        valid = [c for c in candidates if c.fem_error_hz is not None]
        if not valid:
            print("[active] no FEM feedback available; keep previous bounds.")
            return
        best = min(
            valid,
            key=self.fem_joint_score,
        )
        center = np.array(best.params, dtype=float)
        shrink = max(self.args.min_bound_ratio, self.args.bound_shrink ** round_id)
        new_bounds = []
        for value, base in zip(center, self.base_bounds):
            low, high = base
            half_width = (high - low) * shrink / 2.0
            new_low = max(low, float(value - half_width))
            new_high = min(high, float(value + half_width))
            if new_high - new_low < (high - low) * self.args.min_bound_ratio:
                min_half = (high - low) * self.args.min_bound_ratio / 2.0
                new_low = max(low, float(value - min_half))
                new_high = min(high, float(value + min_half))
            new_bounds.append((new_low, new_high))
        self.current_bounds = new_bounds
        print(
            f"[active] best FEM candidate #{best.rank}: error={best.fem_error_hz:.2f} Hz, "
            f"RBG={best.fem_rbg}; next bounds={self.current_bounds}"
        )

    def maybe_retrain(self, round_id: int):
        if not self.args.retrain_command:
            print("[retrain] skipped. Use --retrain-command to attach your model update command.")
            return
        cmd = self.args.retrain_command.format(round=round_id, pool=str(self.work_dir / "active_learning_pool.csv"))
        print("[retrain]", cmd)
        subprocess.run(cmd, shell=True, check=False)

    def run(self):
        for round_id in range(1, self.args.max_rounds + 1):
            candidates = self.run_de(round_id)
            generated_groups = 0
            if self.args.skip_inp:
                print("[inp] skipped by --skip-inp. Candidate png/sector files have been written.")
            else:
                generated_groups = self.generate_inp(round_id)
            if self.args.run_abaqus and not self.args.skip_inp and generated_groups > 0:
                self.run_abaqus(round_id)
            else:
                if generated_groups == 0 and not self.args.skip_inp:
                    print("[abaqus] skipped because no mesh/inp groups were generated.")
                else:
                    print("[abaqus] skipped. Run generated run_abaqus_batch.py manually or pass --run-abaqus.")
            self.parse_feedback(round_id, candidates)
            self.write_round_outputs(round_id, candidates)
            self.print_round_summary(round_id, candidates)
            if self.should_stop(candidates):
                print(f"[stop] target satisfied at round {round_id}.")
                break
            self.update_residual_calibration(candidates, round_id)
            self.update_bounds_from_feedback(candidates, round_id)
            self.maybe_retrain(round_id)

    def should_stop(self, candidates: list[Candidate]) -> bool:
        if not self.args.early_stop:
            return False
        valid = [c for c in candidates if c.status == "fem_valid" and c.fem_error_hz is not None]
        return any(c.fem_error_hz <= self.args.stop_error_hz for c in valid)

    def print_round_summary(self, round_id: int, candidates: list[Candidate]):
        print(f"\n[round {round_id}] summary")
        for c in candidates:
            print(
                f"  #{c.rank:02d} pred={c.pred_freq_hz:.1f}Hz rbg={c.pred_rbg:.4f} "
                f"fem={c.fem_mid_hz} fem_rbg={c.fem_rbg} status={c.status}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Physics-boundary ConvNeXt joint-objective inverse design for phononic-crystal bandgap targets.")
    parser.add_argument("--target-freq", type=float, default=None, help="Target median bandgap frequency in Hz.")
    parser.add_argument(
        "--targets",
        default="",
        help="Comma-separated target frequencies in Hz. Example: 20000,30000,40000,50000,60000,70000",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_DIR,
        help="Root directory for multi-target outputs.",
    )
    parser.add_argument("--min-rbg", type=float, default=0.03, help="Minimum acceptable relative bandgap.")
    parser.add_argument("--freq-error-weight", type=float, default=1.0, help="Weight for center-frequency absolute error in Hz.")
    parser.add_argument("--target-gap-width-hz", type=float, default=0.0, help="Optional desired minimum gap width in Hz; 0 uses min_rbg * target frequency.")
    parser.add_argument("--gap-width-shortfall-weight", type=float, default=1.0, help="Penalty per Hz when predicted gap width is below the desired minimum.")
    parser.add_argument("--gap-width-reward-weight", type=float, default=0.35, help="Reward per useful extra Hz above the desired minimum gap width.")
    parser.add_argument("--gap-width-reward-cap-hz", type=float, default=3000.0, help="Cap useful extra gap width so the search still prioritizes center-frequency accuracy.")
    parser.add_argument(
        "--physics-boundary-ckpt",
        type=Path,
        required=True,
        help="Path to a physics_boundary_convnext checkpoint.",
    )
    parser.add_argument("--rbg-model", type=Path, required=True, help="Path to the relative-bandgap regression model.")
    parser.add_argument("--geometry-filter", type=Path, default=None, help="Optional geometry-validity classifier.")
    parser.add_argument("--rbg-scale", type=float, default=1.0)
    parser.add_argument("--ensemble-std-weight", type=float, default=0.35, help="Penalty multiplier for frequency-model disagreement in Hz.")
    parser.add_argument("--freq-aggregation", choices=["mean", "median", "member"], default="member", help="Frequency surrogate used by DE.")
    parser.add_argument("--freq-member", choices=["physics_boundary_convnext"], default="physics_boundary_convnext", help="Single frequency model used when --freq-aggregation member.")
    parser.add_argument("--penalize-member-disagreement", action="store_true", help="Keep ensemble disagreement penalty even when a single member drives DE.")
    parser.add_argument("--residual-correction", action=argparse.BooleanOptionalAction, default=True, help="Use FEM feedback to shift the next-round surrogate target by pred-FEM residual.")
    parser.add_argument("--residual-step-gain", type=float, default=0.6, help="How much of the best FEM signed error is added to the next surrogate target.")
    parser.add_argument("--max-residual-step-hz", type=float, default=2500.0, help="Maximum per-round change of residual correction.")
    parser.add_argument("--max-residual-correction-hz", type=float, default=9000.0, help="Clamp for automatic FEM residual correction.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Output directory. If omitted, use runs_<target>Hz_physics_boundary_joint to avoid overwriting other targets.",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--de-tol", type=float, default=0.01)
    parser.add_argument("--rbg-penalty", type=float, default=200000.0)
    parser.add_argument("--early-stop", action="store_true", help="Stop before max_rounds when any valid FEM candidate reaches --stop-error-hz. Disabled by default for round-wise SCI reporting.")
    parser.add_argument("--stop-error-hz", type=float, default=800.0)
    parser.add_argument("--prefer-uncertain", action="store_true", help="Add a small exploration bonus toward parameter-space edge regions.")
    parser.add_argument("--uncertain-bonus", type=float, default=300.0)
    parser.add_argument("--adaptive-bounds", action=argparse.BooleanOptionalAction, default=True, help="Shrink next-round DE bounds around the best FEM feedback sample.")
    parser.add_argument("--bound-shrink", type=float, default=0.72, help="Per-round search-range shrink ratio used by adaptive bounds.")
    parser.add_argument("--min-bound-ratio", type=float, default=0.18, help="Minimum bound width ratio relative to the original DE bounds.")
    parser.add_argument("--run-abaqus", action="store_true", help="Actually launch Abaqus batch jobs after inp generation.")
    parser.add_argument("--skip-inp", action="store_true", help="Only run surrogate DE search and save png/sector files; do not call gmsh/inp generation.")
    parser.add_argument(
        "--mesh-python",
        default=str(DEFAULT_ENV3_PYTHON if DEFAULT_ENV3_PYTHON.exists() else Path(sys.executable)),
        help="Python executable used by the isolated gmsh/inp generation subprocess.",
    )
    parser.add_argument("--mesh-timeout", type=int, default=180, help="Best-effort per-candidate mesh timeout in seconds. On Windows in-process gmsh cannot always be interrupted.")
    parser.add_argument("--min-boundary-margin", type=float, default=0.8)
    parser.add_argument("--geometry-legal-threshold", type=float, default=0.65)
    parser.add_argument("--abaqus-cmd", default="abaqus")
    parser.add_argument("--cpus", type=int, default=12)
    parser.add_argument("--retrain-command", default="", help="Optional shell command after each round. Available placeholders: {round}, {pool}.")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.targets:
        targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
        if not targets:
            raise SystemExit("--targets was provided but no valid frequency was parsed.")
        for target in targets:
            run_args = argparse.Namespace(**vars(args))
            run_args.target_freq = target
            run_args.work_dir = None
            tag = ActiveLearningInverseDesigner.target_freq_tag(target)
            suffix = "physics_boundary_joint" if run_args.freq_aggregation == "member" else f"{run_args.freq_aggregation}_joint"
            run_args.work_dir = Path(run_args.output_root) / f"runs_{tag}_{suffix}"
            print("\n" + "=" * 72)
            print(f"[batch] target={target:.2f} Hz -> {run_args.work_dir}")
            print("=" * 72)
            ActiveLearningInverseDesigner(run_args).run()
    else:
        if args.target_freq is None:
            raise SystemExit("Please provide --target-freq for one target or --targets for batch mode.")
        ActiveLearningInverseDesigner(args).run()
