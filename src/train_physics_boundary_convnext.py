from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phonon_dataset import PhononHoleDataset, load_phonon_records


MODEL_NAME = "physics_boundary_convnext"
DEFAULT_IMAGE_DIR = ROOT / "数据处理" / "128x128"
DEFAULT_CSV_PATH = ROOT / "数据处理" / "final_merged_results_updated_v2.csv"


def adapt_first_conv_to_gray(conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        1,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class FixedBoundaryOperators(nn.Module):
    """Fixed differential operators for pore boundary and curvature cues."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("sobel", torch.stack([kx, ky]).unsqueeze(1))
        self.register_buffer("laplacian", lap.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = F.conv2d(x, self.sobel, padding=1)
        edge = torch.sqrt(grad[:, :1].square() + grad[:, 1:].square() + 1e-6)
        curve = F.conv2d(x, self.laplacian, padding=1).abs()
        return edge, curve


class BoundaryEnhancedInput(nn.Module):
    """A light physical prior module before the pretrained ConvNeXt stem.

    The module keeps the image single-channel, so pretrained ConvNeXt weights
    can still be used. Sobel edges highlight impedance/scattering interfaces,
    while the Laplacian response emphasizes local curvature and abrupt changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ops = FixedBoundaryOperators()
        self.fuse = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.gate = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge, curve = self.ops(x)
        edge_n = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        curve_n = curve / curve.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        residual = self.fuse(torch.cat([x, edge_n, curve_n], dim=1))
        enhanced = x + torch.tanh(self.gate) * residual
        return enhanced, edge_n, curve_n


class GeometryDescriptor(nn.Module):
    """Differentiable global descriptors with direct physical interpretation."""

    def __init__(self) -> None:
        super().__init__()
        grid = torch.linspace(-1.0, 1.0, 128)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("xx", xx.clone().contiguous().view(1, 1, 128, 128))
        self.register_buffer("yy", yy.clone().contiguous().view(1, 1, 128, 128))
        self.register_buffer("rr", (xx.square() + yy.square()).sqrt().clone().contiguous().view(1, 1, 128, 128))

    def forward(self, x: torch.Tensor, edge: torch.Tensor, curve: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 128 or x.shape[-2] != 128:
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
            edge = F.interpolate(edge, size=(128, 128), mode="bilinear", align_corners=False)
            curve = F.interpolate(curve, size=(128, 128), mode="bilinear", align_corners=False)

        void = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        mass = void.mean(dim=(2, 3))
        denom = void.sum(dim=(2, 3)).clamp_min(1e-6)
        cx = (void * self.xx).sum(dim=(2, 3)) / denom
        cy = (void * self.yy).sum(dim=(2, 3)) / denom
        radial = (void * self.rr).sum(dim=(2, 3)) / denom
        edge_density = edge.mean(dim=(2, 3))
        curve_density = curve.mean(dim=(2, 3))
        compactness = edge_density.square() / mass.clamp_min(1e-4)
        return torch.cat([mass, edge_density, curve_density, cx, cy, radial, compactness], dim=1)


class PhysicsBoundaryConvNeXt(nn.Module):
    """ConvNeXt-Tiny with a physical boundary module for bandgap regression."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.12,
        use_boundary_enhancement: bool = True,
        use_geometry_descriptors: bool = True,
    ) -> None:
        super().__init__()
        self.use_boundary_enhancement = use_boundary_enhancement
        self.use_geometry_descriptors = use_geometry_descriptors
        self.boundary = BoundaryEnhancedInput()
        self.descriptor = GeometryDescriptor()

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        self.backbone.features[0][0] = adapt_first_conv_to_gray(self.backbone.features[0][0])
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Identity()
        self.norm = nn.LayerNorm(in_features)
        self.geom_norm = nn.LayerNorm(7) if use_geometry_descriptors else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features + (7 if use_geometry_descriptors else 0), 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enh, edge, curve = self.boundary(x)
        backbone_input = x_enh if self.use_boundary_enhancement else x
        feat = self.backbone.features(backbone_input)
        feat = self.backbone.avgpool(feat).flatten(1)
        feat = self.norm(feat)
        if self.use_geometry_descriptors:
            geom = self.descriptor(x, edge, curve)
            feat = torch.cat([feat, self.geom_norm(geom)], dim=1)
        return self.head(feat)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(model_b)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_records(records: list, train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    records = records[:]
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return records[:n_train], records[n_train : n_train + n_val], records[n_train + n_val :]


def make_transforms(image_size: int, train: bool) -> transforms.Compose:
    items = [transforms.Resize((image_size, image_size))]
    if train:
        items.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(90)])
    items.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    return transforms.Compose(items)


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().float().cpu().view(-1)
    target = target.detach().float().cpu().view(-1)
    err = pred - target
    abs_pct_err = err.abs() / target.abs().clamp_min(1e-8)
    mae = err.abs().mean().item()
    mse = err.square().mean().item()
    rmse = math.sqrt(err.square().mean().item())
    mape = abs_pct_err.mean().item() * 100.0
    ss_res = err.square().sum()
    ss_tot = (target - target.mean()).square().sum().clamp_min(1e-8)
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": (1 - ss_res / ss_tot).item()}


def tta_predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    preds = [
        model(x),
        model(torch.flip(x, dims=[3])),
        model(torch.flip(x, dims=[2])),
        model(torch.flip(x, dims=[2, 3])),
    ]
    return torch.stack(preds, dim=0).mean(dim=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler,
    target_mean: float,
    target_std: float,
    ema: EMA | None = None,
    tta_eval: bool = False,
    collect_names: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds_hz, targets_hz, names = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if is_train:
                pred = model(x)
                loss = criterion(pred, y)
            else:
                with torch.no_grad():
                    pred = tta_predict(model, x) if tta_eval else model(x)
                loss = criterion(pred, y)
        if optimizer is not None:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            if ema is not None:
                ema.update(model)
        total_loss += float(loss.detach().float().cpu()) * x.size(0)
        preds_hz.append(pred.detach().float().cpu() * float(target_std) + float(target_mean))
        targets_hz.append(batch["target_hz"].float().cpu())
        if collect_names:
            names.extend(batch["name"])
    pred_all = torch.cat(preds_hz).view(-1)
    target_all = torch.cat(targets_hz).view(-1)
    out = metrics(pred_all, target_all)
    out["loss"] = total_loss / len(loader.dataset)
    if collect_names:
        return out, names, target_all.tolist(), pred_all.tolist()
    return out


def write_predictions(path: Path, names: list[str], y_true: list[float], y_pred: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "true_hz", "pred_hz", "error_hz", "abs_error_hz"])
        for name, true, pred in zip(names, y_true, y_pred):
            err = pred - true
            writer.writerow([name, f"{true:.6f}", f"{pred:.6f}", f"{err:.6f}", f"{abs(err):.6f}"])


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    records = load_phonon_records(args.csv_path, args.image_dir, 0, args.target_col)
    train_records, val_records, test_records = split_records(records, args.train_ratio, args.val_ratio, args.seed)

    train_ds = PhononHoleDataset(train_records, make_transforms(args.image_size, True))
    train_eval_ds = PhononHoleDataset(train_records, make_transforms(args.image_size, False), train_ds.target_mean, train_ds.target_std)
    val_ds = PhononHoleDataset(val_records, make_transforms(args.image_size, False), train_ds.target_mean, train_ds.target_std)
    test_ds = PhononHoleDataset(test_records, make_transforms(args.image_size, False), train_ds.target_mean, train_ds.target_std)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    variant_flags = {
        "full": (True, True),
        "boundary_only": (True, False),
        "geometry_only": (False, True),
        "plain": (False, False),
    }
    use_boundary_enhancement, use_geometry_descriptors = variant_flags[args.variant]
    model = PhysicsBoundaryConvNeXt(
        pretrained=args.pretrained,
        dropout=args.dropout,
        use_boundary_enhancement=use_boundary_enhancement,
        use_geometry_descriptors=use_geometry_descriptors,
    ).to(device)
    ema = EMA(model, args.ema_decay) if args.ema else None
    criterion = nn.SmoothL1Loss(beta=args.smooth_l1_beta)

    head_params = list(model.head.parameters()) + list(model.norm.parameters()) + list(model.geom_norm.parameters())
    prior_params = list(model.boundary.parameters()) if use_boundary_enhancement else []
    special_ids = {id(p) for p in head_params + prior_params}
    backbone_params = [p for p in model.parameters() if id(p) not in special_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
            {"params": prior_params, "lr": args.lr * args.prior_lr_scale},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and args.amp else None

    run_dir = args.output_dir / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    bad_epochs = 0
    history = []
    print(f"[train] {MODEL_NAME}/{args.variant}: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test on {device}")
    print(f"[train] pretrained={args.pretrained} ema={args.ema} tta_eval={args.tta_eval} boundary={use_boundary_enhancement} descriptors={use_geometry_descriptors}")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train_ds.target_mean, train_ds.target_std, ema)
        eval_model = ema.shadow if ema is not None else model
        val_m = run_epoch(eval_model, val_loader, criterion, None, device, None, train_ds.target_mean, train_ds.target_std, tta_eval=args.tta_eval)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_m, "val": val_m, "lr": scheduler.get_last_lr()[0]})
        print(f"epoch {epoch:03d} | train MAE {train_m['mae']:.2f} Hz | val MAE {val_m['mae']:.2f} Hz | val R2 {val_m['r2']:.4f}")
        if val_m["mae"] < best_val:
            best_val = val_m["mae"]
            bad_epochs = 0
            save_model = ema.shadow if ema is not None else model
            torch.save(
                {
                    "model": save_model.state_dict(),
                    "args": vars(args),
                    "target_mean": train_ds.target_mean,
                    "target_std": train_ds.target_std,
                    "model_name": MODEL_NAME,
                },
                run_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[early-stop] no validation MAE improvement for {args.patience} epochs.")
                break

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    train_m_final, train_names, train_y_true, train_y_pred = run_epoch(
        model,
        train_eval_loader,
        criterion,
        None,
        device,
        None,
        train_ds.target_mean,
        train_ds.target_std,
        tta_eval=args.tta_eval,
        collect_names=True,
    )
    test_m, test_names, test_y_true, test_y_pred = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        device,
        None,
        train_ds.target_mean,
        train_ds.target_std,
        tta_eval=args.tta_eval,
        collect_names=True,
    )
    summary = {
        "model": MODEL_NAME,
        "variant": args.variant,
        "pretrained": args.pretrained,
        "best_val_mae": best_val,
        "train": train_m_final,
        "test": test_m,
        "physical_modules": [
            *(["Sobel boundary", "Laplacian curvature"] if use_boundary_enhancement else []),
            *(["geometry descriptors"] if use_geometry_descriptors else []),
        ],
    }
    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_predictions(run_dir / "train_predictions.csv", train_names, train_y_true, train_y_pred)
    write_predictions(run_dir / "test_predictions.csv", test_names, test_y_true, test_y_pred)
    print(
        f"[train-final] MAE {train_m_final['mae']:.2f} Hz | MSE {train_m_final['mse']:.2f} | "
        f"RMSE {train_m_final['rmse']:.2f} Hz | MAPE {train_m_final['mape']:.4f}% | R2 {train_m_final['r2']:.4f}"
    )
    print(
        f"[test] MAE {test_m['mae']:.2f} Hz | MSE {test_m['mse']:.2f} | "
        f"RMSE {test_m['rmse']:.2f} Hz | MAPE {test_m['mape']:.4f}% | R2 {test_m['r2']:.4f}"
    )
    print(f"[done] saved to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a physical-boundary-enhanced ConvNeXt regressor.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument(
        "--variant",
        choices=["full", "boundary_only", "geometry_only", "plain"],
        default="full",
        help="Ablation configuration with a shared ConvNeXt backbone and regression head.",
    )
    parser.add_argument("--target-col", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.2)
    parser.add_argument("--prior-lr-scale", type=float, default=0.8)
    parser.add_argument("--weight-decay", type=float, default=8e-5)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--device", default="")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--tta-eval", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

