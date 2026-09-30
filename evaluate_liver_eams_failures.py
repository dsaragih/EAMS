"""Evaluate liver EAMS experiments and summarize per-mesh segmentation failures.

This is intentionally separate from ``test_experiment.py``.  It reuses its
saved-config/checkpoint loading path, evaluates the best checkpoint in every
selected fold, and then analyzes the resulting per-mesh ``.pt`` artifacts.

Example:
    python evaluate_liver_eams_failures.py /path/to/experiment_a /path/to/experiment_b
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr

from data.features.edge_features import compute_dihedrals
from data.features.node_features import (
    compute_principal_curvatures,
    compute_stable_pca_frame_diagnostics,
)
from test_lightning import run_test
from util.config import register_custom_omegaconf_resolvers
from util.experiment_test_utils import clone_cfg, find_best_checkpoint, load_test_cfg_from_experiment_dir


EPS = 1e-6
FOREGROUND_CLASS_IDS = (1, 2)
CLASS_NAMES = {0: "background", 1: "ligament", 2: "ridge"}
OUTCOMES = ("foreground_dice", "ligament_dice", "ridge_dice")
CORRELATION_EXCLUDED_FEATURES = {
    "feature_pca_eigenvalue_1",
    "feature_pca_eigenvalue_2",
    "feature_pca_eigenvalue_3",
}
CORRELATION_EXCLUDED_PREFIXES = ("feature_prediction_",)


class UndirectedEdgeCollapseError(ValueError):
    """A mesh cannot be reduced to physical edges without losing label data."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", help="One or more liver experiment directories.")
    parser.add_argument("--fold", type=int, default=None, help="Optional 0-based fold index.")
    parser.add_argument("--results-subdir", default="liver_failure_analysis")
    parser.add_argument("--reuse-existing-results", action="store_true", help="Skip inference and analyze existing prediction artifacts.")
    parser.add_argument("--failure-examples", type=int, default=6, help="Lowest-Dice examples rendered per experiment.")
    parser.add_argument("--reference-examples", type=int, default=2, help="Highest-Dice examples rendered per experiment.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Combined-report directory; defaults under the first experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Validate experiment/checkpoint selection without inference or writes.")
    return parser.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "experiment"


def _best_checkpoint_paths(checkpoints_dir: Path, fold: int | None) -> list[tuple[int, Path]]:
    fold_dirs = []
    for candidate in checkpoints_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith("fold_"):
            continue
        suffix = candidate.name.removeprefix("fold_")
        if suffix.isdigit() and (fold is None or int(suffix) == fold):
            fold_dirs.append((int(suffix), candidate))
    fold_dirs.sort()
    if fold is not None and not fold_dirs:
        raise FileNotFoundError(f"No fold_{fold} found under {checkpoints_dir}")
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {checkpoints_dir}")
    return [(fold_idx, Path(find_best_checkpoint(str(fold_dir)))) for fold_idx, fold_dir in fold_dirs]


def _condition_results_dir(base_dir: Path, cfg: DictConfig) -> Path:
    condition = cfg.transforms.get("condition_name", None)
    return base_dir / str(condition) if condition else base_dir


def _run_experiment(
    experiment_dir: str,
    fold: int | None,
    results_subdir: str,
    reuse_existing: bool,
    dry_run: bool,
) -> tuple[str, Path, list[Path]]:
    cfg, artifacts, _ = load_test_cfg_from_experiment_dir(experiment_dir)
    if "liver" not in str(cfg.data.get("_target_", "")).lower():
        raise ValueError(f"{experiment_dir} is not a liver experiment.")

    experiment_id = _slug(str(artifacts["root"].name))
    analysis_root = artifacts["results_dir"] / results_subdir
    checkpoint_paths = _best_checkpoint_paths(artifacts["checkpoints_dir"], fold)
    result_dirs: list[Path] = []
    for fold_idx, checkpoint_path in checkpoint_paths:
        base_dir = analysis_root / "predictions" / f"fold_{fold_idx}" / checkpoint_path.stem
        checkpoint_cfg = clone_cfg(cfg)
        checkpoint_cfg.experiment_dir = str(artifacts["root"])
        checkpoint_cfg.checkpoint_path = str(checkpoint_path)
        checkpoint_cfg.paths.results_dir = str(base_dir)
        checkpoint_cfg.model.save_pt_results = True
        if "current_fold" in checkpoint_cfg.data:
            checkpoint_cfg.data.current_fold = fold_idx
        result_dir = _condition_results_dir(base_dir, checkpoint_cfg)
        result_dirs.append(result_dir)
        print(f"{experiment_id}: fold {fold_idx}, checkpoint {checkpoint_path.name}")
        print(f"  prediction artifacts: {result_dir}")
        if not reuse_existing and not dry_run:
            run_test(checkpoint_cfg)
    return experiment_id, analysis_root, result_dirs


def _as_cpu_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Saved result is missing tensor '{name}'.")
    return value.detach().cpu()


def collapse_undirected_edges(
    edge_index: torch.Tensor,
    edge_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse directed edges while requiring consistent labels per physical edge."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E).")
    if edge_index.shape[1] != edge_labels.numel():
        raise ValueError("edge labels must align with edge_index columns.")
    canonical = torch.sort(edge_index.long(), dim=0).values.t().contiguous()
    unique_edges, inverse = torch.unique(canonical, dim=0, return_inverse=True)
    labels = torch.full((unique_edges.shape[0],), -1, dtype=edge_labels.dtype)
    for edge_idx, label in enumerate(edge_labels.tolist()):
        group = int(inverse[edge_idx])
        existing = int(labels[group])
        if existing not in (-1, int(label)):
            raise UndirectedEdgeCollapseError(
                "Inconsistent labels for opposite directed edges."
            )
        labels[group] = int(label)
    return unique_edges.t().contiguous(), labels


def mean_logits_over_undirected_edges(
    edge_index: torch.Tensor,
    edge_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one physical-edge prediction from the mean of directed logits.

    Liver test Dice remains directed-edge Dice.  This reduction is only for a
    single unambiguous class in visualizations, where the two directions can
    legitimately receive different model predictions.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E).")
    if edge_logits.ndim != 2 or edge_logits.shape[0] != edge_index.shape[1]:
        raise ValueError("edge logits must have shape (E, C) aligned with edge_index.")
    canonical = torch.sort(edge_index.long(), dim=0).values.t().contiguous()
    unique_edges, inverse = torch.unique(canonical, dim=0, return_inverse=True)
    mean_logits = edge_logits.new_zeros((unique_edges.shape[0], edge_logits.shape[1]))
    mean_logits.index_add_(0, inverse, edge_logits)
    counts = torch.bincount(inverse, minlength=unique_edges.shape[0]).to(edge_logits)
    mean_logits = mean_logits / counts.unsqueeze(1).clamp_min(1)
    return unique_edges.t().contiguous(), mean_logits.argmax(dim=1)


def _edge_lengths(pos: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pos[edges[0]] - pos[edges[1]], dim=-1)


def _dice(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> float:
    pred_mask = pred == class_id
    target_mask = target == class_id
    intersection = (pred_mask & target_mask).sum().item()
    return float((2.0 * intersection + EPS) / (pred_mask.sum().item() + target_mask.sum().item() + EPS))


def _class_error_rates(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> tuple[float, float]:
    truth = target == class_id
    predicted = pred == class_id
    fnr = float((truth & ~predicted).sum().item() / max(int(truth.sum()), 1))
    fpr = float((~truth & predicted).sum().item() / max(int((~truth).sum()), 1))
    return fnr, fpr


def _edge_component_count(edges: torch.Tensor) -> int:
    if edges.numel() == 0:
        return 0
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in edges.t().tolist():
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[int] = set()
    components = 0
    for start in adjacency:
        if start in seen:
            continue
        components += 1
        queue = deque([start])
        seen.add(start)
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
    return components


def _class_vertex_masks(
    num_nodes: int,
    edges: torch.Tensor,
    labels: torch.Tensor,
    class_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(num_nodes, dtype=torch.bool)
    has_valid = torch.zeros(num_nodes, dtype=torch.bool)
    has_foreground = torch.zeros(num_nodes, dtype=torch.bool)
    valid = labels >= 0
    for mask, predicate in ((target, labels == class_id), (has_valid, valid), (has_foreground, labels > 0)):
        selected = edges[:, predicate]
        if selected.numel():
            mask[selected.flatten()] = True
    background = has_valid & ~has_foreground
    return target, background


def _finite_mean(values: torch.Tensor) -> float:
    return float(values.float().mean()) if values.numel() else float("nan")


def _finite_median(values: torch.Tensor) -> float:
    return float(values.float().median()) if values.numel() else float("nan")


def _surface_area(data: Any, pos: torch.Tensor, face: torch.Tensor) -> float:
    cached = getattr(data, "face_area", None)
    if isinstance(cached, torch.Tensor) and cached.numel() == face.shape[1]:
        return float(cached.sum())
    v0, v1, v2 = pos[face[0]], pos[face[1]], pos[face[2]]
    return float((0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)).sum())


def _stable_pca_features(data: Any) -> dict[str, float]:
    diagnostics = compute_stable_pca_frame_diagnostics(data)
    eigenvalues = diagnostics["eigenvalues"].detach().cpu()
    gaps = diagnostics["relative_eigengaps"].detach().cpu()
    margins = diagnostics["sign_margins"].detach().cpu()
    return {
        "feature_pca_eigenvalue_1": float(eigenvalues[0]),
        "feature_pca_eigenvalue_2": float(eigenvalues[1]),
        "feature_pca_eigenvalue_3": float(eigenvalues[2]),
        "feature_pca_eigengap_12": float(gaps[0]),
        "feature_pca_eigengap_23": float(gaps[1]),
        "feature_pca_sign_margin_1": float(margins[0]),
        "feature_pca_sign_margin_2": float(margins[1]),
    }


def analyze_saved_mesh(path: Path, experiment_id: str, fold: int) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    pos = _as_cpu_tensor(getattr(data, "pos", None), "pos").float()
    face = _as_cpu_tensor(getattr(data, "face", None), "face").long()
    edge_index = _as_cpu_tensor(getattr(data, "edge_index", None), "edge_index").long()
    target = _as_cpu_tensor(getattr(data, "edge_labels", None), "edge_labels").long()
    pred = _as_cpu_tensor(getattr(data, "pred_edge_labels", None), "pred_edge_labels").long()
    logits = _as_cpu_tensor(getattr(data, "pred_edge_logits", None), "pred_edge_logits").float()
    if pred.numel() != target.numel() or logits.shape[0] != target.numel():
        raise ValueError(f"Prediction tensors do not align in {path}")

    valid = target >= 0
    pred_valid, target_valid = pred[valid], target[valid]
    undirected_edges, undirected_labels = collapse_undirected_edges(edge_index[:, valid], target_valid)
    edge_lengths = _edge_lengths(pos, undirected_edges)
    median_length = edge_lengths.median().clamp_min(EPS)
    surface_area = _surface_area(data, pos, face)
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "fold": fold,
        "mesh_id": path.stem,
        "artifact_path": str(path),
        "foreground_dice": float(np.mean([_dice(pred_valid, target_valid, class_id) for class_id in FOREGROUND_CLASS_IDS])),
        "ligament_dice": _dice(pred_valid, target_valid, 1),
        "ridge_dice": _dice(pred_valid, target_valid, 2),
        "feature_num_vertices": int(pos.shape[0]),
        "feature_num_faces": int(face.shape[1]),
        "feature_num_directed_edges": int(edge_index.shape[1]),
        "feature_num_undirected_edges": int(undirected_edges.shape[1]),
        "feature_surface_area": surface_area,
        "feature_normalized_edge_length_mean": float((edge_lengths / median_length).mean()),
        "feature_normalized_edge_length_p90": float(torch.quantile(edge_lengths / median_length, 0.9)),
        **_stable_pca_features(data),
    }

    dihedrals = compute_dihedrals(data).detach().cpu()[valid]
    _, inverse = torch.unique(torch.sort(edge_index[:, valid], dim=0).values.t(), dim=0, return_inverse=True)
    unique_dihedrals = torch.empty(undirected_edges.shape[1], dtype=dihedrals.dtype)
    for idx in range(unique_dihedrals.numel()):
        unique_dihedrals[idx] = dihedrals[inverse == idx].mean()
    curvature = compute_principal_curvatures(data).detach().cpu().abs().max(dim=1).values

    for class_id, class_name in CLASS_NAMES.items():
        if class_id == 0:
            continue
        class_mask = undirected_labels == class_id
        fraction = float(class_mask.float().mean()) if undirected_labels.numel() else 0.0
        target_vertices, background_vertices = _class_vertex_masks(pos.shape[0], undirected_edges, undirected_labels, class_id)
        fnr, fpr = _class_error_rates(pred_valid, target_valid, class_id)
        prefix = f"{class_name}"
        row.update(
            {
                f"feature_{prefix}_edge_fraction": fraction,
                f"feature_{prefix}_edge_count": int(class_mask.sum()),
                f"feature_{prefix}_components": _edge_component_count(undirected_edges[:, class_mask]),
                f"feature_{prefix}_normalized_edge_length_mean": _finite_mean(edge_lengths[class_mask] / median_length),
                f"feature_{prefix}_normalized_edge_length_median": _finite_median(edge_lengths[class_mask] / median_length),
                f"feature_{prefix}_dihedral_deviation_mean": _finite_mean((math.pi - unique_dihedrals[class_mask]).abs()),
                f"feature_{prefix}_curvature_mean": _finite_mean(curvature[target_vertices]),
                f"feature_{prefix}_curvature_background_mean": _finite_mean(curvature[background_vertices]),
                f"feature_{prefix}_curvature_contrast": _finite_mean(curvature[target_vertices]) - _finite_mean(curvature[background_vertices]),
                f"error_{prefix}_false_negative_rate": fnr,
                f"error_{prefix}_false_positive_rate": fpr,
            }
        )

    probabilities = torch.softmax(logits[valid], dim=-1)
    sorted_probabilities = probabilities.topk(k=min(2, probabilities.shape[1]), dim=-1).values
    margin = sorted_probabilities[:, 0] - sorted_probabilities[:, 1] if sorted_probabilities.shape[1] == 2 else torch.ones_like(sorted_probabilities[:, 0])
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum(dim=-1)
    correct = pred_valid == target_valid
    row.update(
        {
            "feature_prediction_margin_mean": _finite_mean(margin),
            "feature_prediction_entropy_mean": _finite_mean(entropy),
            "feature_prediction_margin_correct_mean": _finite_mean(margin[correct]),
            "feature_prediction_margin_incorrect_mean": _finite_mean(margin[~correct]),
            "error_edge_misclassification_rate": float((~correct).float().mean()),
        }
    )
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    descriptors = (
        sorted(
            key
            for key in rows[0]
            if key.startswith("feature_")
            and key not in CORRELATION_EXCLUDED_FEATURES
            and not key.startswith(CORRELATION_EXCLUDED_PREFIXES)
        )
        if rows
        else []
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment_id"]), int(row["fold"]))].append(row)
    results = []
    for (experiment_id, fold), group in sorted(grouped.items()):
        for outcome in OUTCOMES:
            for descriptor in descriptors:
                pairs = [
                    (float(row[descriptor]), float(row[outcome]))
                    for row in group
                    if isinstance(row.get(descriptor), (int, float))
                    and math.isfinite(float(row[descriptor]))
                    and math.isfinite(float(row[outcome]))
                ]
                if len(pairs) < 3 or len({pair[0] for pair in pairs}) < 2:
                    rho, p_value = float("nan"), float("nan")
                else:
                    statistic = spearmanr(*zip(*pairs))
                    rho, p_value = float(statistic.statistic), float(statistic.pvalue)
                results.append({"experiment_id": experiment_id, "fold": fold, "outcome": outcome, "descriptor": descriptor, "n": len(pairs), "spearman_rho": rho, "p_value": p_value})
    return results


def correlation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if math.isfinite(float(row["spearman_rho"])):
            grouped[(row["outcome"], row["descriptor"])].append(row)
    summary = []
    for (outcome, descriptor), items in sorted(grouped.items()):
        rhos = [float(item["spearman_rho"]) for item in items]
        median = float(np.median(rhos))
        consistency = float(np.mean([math.copysign(1.0, rho) == math.copysign(1.0, median) for rho in rhos]))
        summary.append({"outcome": outcome, "descriptor": descriptor, "median_spearman_rho": median, "direction_consistency": consistency, "consistent_association": bool(len(items) >= 2 and abs(median) >= 0.3 and consistency >= 0.75)})
    return summary


def _render_edges(axis, pos: np.ndarray, edges: torch.Tensor, labels: torch.Tensor, title: str, error: bool = False) -> None:
    axis.set_title(title, fontsize=9)
    colors = {1: "#27a6a4", 2: "#dc5a5a"}
    if error:
        colors = {1: "#ff8c00", 2: "#d62728", 3: "#8a2be2", 4: "#1f77b4"}
    for edge, label in zip(edges.t().tolist(), labels.tolist()):
        label = int(label)
        if label == 0:
            continue
        points = pos[edge]
        axis.plot(points[:, 0], points[:, 1], points[:, 2], color=colors.get(label, "#111111"), linewidth=0.7)
    axis.set_axis_off()


def render_example(row: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = torch.load(row["artifact_path"], map_location="cpu", weights_only=False)
    pos = _as_cpu_tensor(getattr(data, "pos", None), "pos").numpy()
    face = _as_cpu_tensor(getattr(data, "face", None), "face").numpy().T
    edges, target = collapse_undirected_edges(data.edge_index.cpu(), data.edge_labels.cpu())
    prediction_edges, predicted = mean_logits_over_undirected_edges(
        data.edge_index.cpu(), data.pred_edge_logits.cpu()
    )
    if not torch.equal(edges, prediction_edges):
        raise RuntimeError("Ground-truth and prediction edge reductions disagree.")
    error_labels = torch.zeros_like(target)
    error_labels[(target == 1) & (predicted != 1)] = 1
    error_labels[(target == 2) & (predicted != 2)] = 2
    error_labels[(target != 1) & (predicted == 1)] = 3
    error_labels[(target != 2) & (predicted == 2)] = 4
    figure = plt.figure(figsize=(12, 4))
    for index, (labels, title, error) in enumerate(((target, "Ground truth", False), (predicted, "Prediction", False), (error_labels, "Errors: FN lig/ridge, FP lig/ridge", True)), start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.add_collection3d(Poly3DCollection(pos[face], facecolors="#e8e8e8", edgecolors="none", alpha=0.25))
        _render_edges(axis, pos, edges, labels, title, error)
        axis.auto_scale_xyz(pos[:, 0], pos[:, 1], pos[:, 2])
        axis.view_init(elev=20, azim=-55)
    figure.suptitle(f"{row['mesh_id']} | fold {row['fold']} | foreground Dice={row['foreground_dice']:.3f} | lig/ridge={row['ligament_dice']:.3f}/{row['ridge_dice']:.3f}", fontsize=10)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_selected_examples(
    rows: list[dict[str, Any]],
    output_dir: Path,
    failures: int,
    references: int,
) -> list[dict[str, str]]:
    """Render enough examples, replacing artifacts that cannot be collapsed."""
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["foreground_dice"]),
            -float(row["error_edge_misclassification_rate"]),
        ),
    )
    candidates = [("failure", row) for row in ordered] + [
        ("reference", row) for row in reversed(ordered)
    ]
    remaining = {"failure": failures, "reference": references}
    skipped = []
    for role, row in candidates:
        if remaining[role] <= 0:
            continue
        image_name = f"{role}_{row['fold']}_{_slug(str(row['mesh_id']))}.png"
        try:
            render_example(row, output_dir / image_name)
        except UndirectedEdgeCollapseError as error:
            skipped.append(
                {
                    "artifact_path": str(row["artifact_path"]),
                    "stage": "render",
                    "reason": str(error),
                }
            )
            print(f"Skipping render for {row['mesh_id']}: {error}")
            continue
        remaining[role] -= 1
        if not any(remaining.values()):
            break
    return skipped


def _write_markdown(summary: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path) -> None:
    by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in summary:
        if item["consistent_association"]:
            by_outcome[str(item["outcome"])].append(item)
    lines = ["# Liver EAMS failure analysis", "", f"{len(rows)} model-instance/mesh observations across {len({row['experiment_id'] for row in rows})} experiment(s).", ""]
    for outcome in OUTCOMES:
        lines.extend([f"## {outcome}", "", "| Descriptor | Median rho | Direction consistency |", "| --- | ---: | ---: |"])
        associations = sorted(by_outcome[outcome], key=lambda item: abs(float(item["median_spearman_rho"])), reverse=True)
        if not associations:
            lines.append("| No descriptor passed the consistency threshold | n/a | n/a |")
        for item in associations[:10]:
            lines.append(f"| {item['descriptor']} | {item['median_spearman_rho']:.3f} | {item['direction_consistency']:.2f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _fold_from_result_dir(path: Path) -> int:
    for parent in path.parents:
        if parent.name.startswith("fold_") and parent.name.removeprefix("fold_").isdigit():
            return int(parent.name.removeprefix("fold_"))
    raise ValueError(f"Could not infer fold from {path}")


def main() -> None:
    args = _parse_args()
    if args.failure_examples < 0 or args.reference_examples < 0:
        raise ValueError("Example counts must be non-negative.")
    register_custom_omegaconf_resolvers()
    experiment_runs = [
        _run_experiment(path, args.fold, args.results_subdir, args.reuse_existing_results, args.dry_run)
        for path in args.experiment_dirs
    ]
    if args.dry_run:
        print("Dry run complete; no inference or reports were written.")
        return

    all_rows: list[dict[str, Any]] = []
    roots: dict[str, Path] = {}
    for experiment_id, analysis_root, result_dirs in experiment_runs:
        roots[experiment_id] = analysis_root
        experiment_rows = []
        skipped = []
        for result_dir in result_dirs:
            fold = _fold_from_result_dir(result_dir)
            artifacts = sorted(result_dir.glob("*.pt"))
            if not artifacts:
                raise FileNotFoundError(f"No .pt prediction artifacts found in {result_dir}; run without --reuse-existing-results first.")
            for path in artifacts:
                try:
                    experiment_rows.append(analyze_saved_mesh(path, experiment_id, fold))
                except UndirectedEdgeCollapseError as error:
                    skipped.append(
                        {
                            "artifact_path": str(path),
                            "stage": "analysis",
                            "reason": str(error),
                        }
                    )
                    print(f"Skipping {path.name}: {error}")
        if not experiment_rows:
            raise RuntimeError(f"No analyzable prediction artifacts found for {experiment_id}.")
        _write_csv(experiment_rows, analysis_root / "per_mesh_failure_features.csv")
        experiment_correlations = correlation_rows(experiment_rows)
        experiment_summary = correlation_summary(experiment_correlations)
        _write_csv(experiment_correlations, analysis_root / "correlations_by_fold.csv")
        _write_csv(experiment_summary, analysis_root / "correlation_summary.csv")
        (analysis_root / "correlation_summary.json").write_text(
            json.dumps(experiment_summary, indent=2) + "\n"
        )
        _write_markdown(experiment_summary, experiment_rows, analysis_root / "failure_summary.md")
        skipped.extend(
            _render_selected_examples(
                experiment_rows,
                analysis_root / "examples",
                args.failure_examples,
                args.reference_examples,
            )
        )
        _write_csv(skipped, analysis_root / "skipped_artifacts.csv")
        all_rows.extend(experiment_rows)

    correlations = correlation_rows(all_rows)
    summary = correlation_summary(correlations)
    combined_root = args.output_dir or roots[experiment_runs[0][0]] / "cross_experiment"
    combined_root.mkdir(parents=True, exist_ok=True)
    _write_csv(all_rows, combined_root / "per_mesh_failure_features.csv")
    _write_csv(correlations, combined_root / "correlations_by_fold.csv")
    _write_csv(summary, combined_root / "correlation_summary.csv")
    (combined_root / "correlation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, all_rows, combined_root / "failure_summary.md")
    print(f"Wrote combined failure analysis to: {combined_root}")


if __name__ == "__main__":
    main()
