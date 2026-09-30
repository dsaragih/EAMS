#!/usr/bin/env python3
"""Measure PCA-frame and prediction stability on Liver or Teeth3DS meshes.

This is intentionally a standalone evaluation script: it restores each best
fold checkpoint from an existing experiment and never changes its config or
checkpoints.  It writes one CSV row per mesh/rotation and a compact JSON
summary, including the largest prediction disagreements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any
from tqdm import tqdm

import hydra
import torch
from torch_geometric.data import Batch

from data.features.node_features import (
    compute_dental_pca_frame_diagnostics,
    compute_stable_pca_frame_diagnostics,
)
from data.transforms.transforms import _apply_rigid_transform
from util.config import (
    EMNN_COORDINATE_NODE_FEATURES,
    FEATURE_SIZES,
    register_custom_omegaconf_resolvers,
)
from util.experiment_test_utils import (
    clone_cfg,
    discover_checkpoint_paths,
    find_best_checkpoint,
    load_test_cfg_from_experiment_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--num-rotations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--top-disagreements", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <experiment>/results/pca_stability.",
    )
    return parser.parse_args()


def _device_from_arg(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_runtime_config(cfg):
    cfg.data.transforms = clone_cfg(cfg.transforms)
    cfg.data.num_workers = 0
    cfg.data.serial_batches = True
    return cfg


def _best_checkpoints(checkpoints_dir: Path, fold: int | None) -> list[tuple[int, Path]]:
    by_fold: dict[int, Path] = {}
    for fold_idx, checkpoint in discover_checkpoint_paths(checkpoints_dir, fold=fold):
        by_fold.setdefault(fold_idx, checkpoint.parent)
    return [
        (fold_idx, Path(find_best_checkpoint(str(fold_dir))))
        for fold_idx, fold_dir in sorted(by_fold.items())
    ]


def _load_model(cfg, checkpoint_path: Path, device: torch.device):
    model_class = hydra.utils.get_class(cfg.model._target_)
    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            results_dir=str(cfg.paths.results_dir),
            weights_only=False,
        )
    except Exception:
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            results_dir=str(cfg.paths.results_dir),
            network_config=cfg.model.network_config,
            metrics_dict=cfg.model.metrics_dict,
            weights_only=False,
        )
    return model.to(device).eval()


def _mesh_id(data) -> str:
    for key in ("mesh_id", "filename", "name"):
        value = getattr(data, key, None)
        if isinstance(value, str) and value:
            return value
    return f"mesh_{int(data.num_nodes)}_nodes"


def _axis_angle_rotation(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, float]:
    axis = torch.randn(3, generator=generator, dtype=torch.float32)
    axis = axis / torch.linalg.norm(axis).clamp_min(1e-12)
    angle = float(torch.rand((), generator=generator).item() * 2.0 * torch.pi)
    x, y, z = axis
    zero = torch.zeros((), dtype=torch.float32)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    matrix = (
        torch.eye(3, dtype=torch.float32)
        + torch.sin(torch.tensor(angle)) * skew
        + (1.0 - torch.cos(torch.tensor(angle))) * (skew @ skew)
    )
    return matrix, axis, angle


@torch.inference_mode()
def _predict_probabilities(model, data, device: torch.device) -> torch.Tensor:
    probabilities, _, _ = _predict_with_diagnostics(model, data, device)
    return probabilities


@torch.inference_mode()
def _predict_with_diagnostics(model, data, device: torch.device):
    batch = Batch.from_data_list([data.clone()]).to(device)
    batch = model.on_after_batch_transfer(batch, dataloader_idx=0)
    output = model(batch)
    # Teeth3DS supervision is face-level. Its standard EMNN decoder emits
    # vertex logits, which the Lightning module pools to faces for loss and
    # metrics; use that same prediction unit for disagreement measurements.
    if hasattr(model, "_get_face_logits") and hasattr(batch, "face_labels"):
        logits = model._get_face_logits(batch, output)
    else:
        logits = output["pred"]
    model_inputs = {
        key: getattr(batch, key).detach().cpu()
        for key in ("x", "edge_attr", "face_attr", "pos", "normal")
        if isinstance(getattr(batch, key, None), torch.Tensor)
    }
    model_outputs = {
        key: value.detach().cpu()
        for key, value in {
            "raw_pred": output.get("pred"),
            "embedding": output.get("embedding"),
        }.items()
        if isinstance(value, torch.Tensor)
    }
    return torch.softmax(logits, dim=-1).detach().cpu(), model_inputs, model_outputs


def _difference_stats(base: torch.Tensor | None, rotated: torch.Tensor | None) -> dict[str, float]:
    if base is None or rotated is None:
        return {"mean": float("nan"), "max": float("nan")}
    if base.shape != rotated.shape:
        raise ValueError(
            f"Diagnostic tensor shape changed: {tuple(base.shape)} -> {tuple(rotated.shape)}"
        )
    difference = (rotated - base).abs()
    return {"mean": float(difference.mean()), "max": float(difference.max())}


def _equivariant_difference_stats(
    base: torch.Tensor | None,
    rotated: torch.Tensor | None,
    matrix: torch.Tensor,
) -> dict[str, float]:
    if base is None or rotated is None:
        return {"mean": float("nan"), "max": float("nan")}
    expected = base @ matrix.t()
    return _difference_stats(expected, rotated)


def _top_two_margin(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[-1] < 2:
        return torch.ones(probabilities.shape[0], dtype=probabilities.dtype)
    top_two = probabilities.topk(k=2, dim=-1).values
    return top_two[:, 0] - top_two[:, 1]


def _feature_key(feature_name: str) -> str:
    return feature_name


def _node_feature_slices(cfg) -> list[tuple[str, slice]]:
    offset = 0
    slices = []
    for feature_name in cfg.features.get("node_features") or []:
        if feature_name in EMNN_COORDINATE_NODE_FEATURES:
            continue
        width = FEATURE_SIZES[feature_name]
        slices.append((feature_name, slice(offset, offset + width)))
        offset += width
    return slices


def _node_feature_difference_diagnostics(
    base_x: torch.Tensor | None,
    rotated_x: torch.Tensor | None,
    feature_slices: list[tuple[str, slice]],
) -> dict[str, Any]:
    """Report device-featurised scalar-input drift by configured feature group."""
    if base_x is None or rotated_x is None:
        return {}

    details: dict[str, Any] = {}
    largest_feature = ""
    largest_component = ""
    largest_maximum = -1.0
    for feature_name, feature_slice in feature_slices:
        base_feature = base_x[:, feature_slice]
        rotated_feature = rotated_x[:, feature_slice]
        difference = (rotated_feature - base_feature).abs()
        key = _feature_key(feature_name)
        feature_maximum = float(difference.max())
        details[f"input_feature_{key}_mean_abs_difference"] = float(difference.mean())
        details[f"input_feature_{key}_max_abs_difference"] = feature_maximum
        if feature_maximum > largest_maximum:
            largest_feature = feature_name
            largest_component = feature_name
            largest_maximum = feature_maximum

        if not feature_name.endswith("_cylindrical"):
            continue
        for component_idx, component_name in enumerate(("radius", "theta", "z")):
            base_component = base_feature[:, component_idx]
            rotated_component = rotated_feature[:, component_idx]
            component_difference = (rotated_component - base_component).abs()
            component_key = f"input_feature_{key}_{component_name}"
            component_maximum = float(component_difference.max())
            details[f"{component_key}_mean_abs_difference"] = float(
                component_difference.mean()
            )
            details[f"{component_key}_max_abs_difference"] = component_maximum
            if component_maximum >= largest_maximum:
                largest_feature = feature_name
                largest_component = component_name
                largest_maximum = component_maximum

            if component_name == "z":
                nonzero = (base_component.abs() > 1e-6) & (
                    rotated_component.abs() > 1e-6
                )
                sign_disagreement = (
                    base_component[nonzero].sign()
                    != rotated_component[nonzero].sign()
                )
                details[f"{component_key}_sign_disagreement_fraction"] = float(
                    sign_disagreement.float().mean() if sign_disagreement.numel() else 0.0
                )
                details[f"{component_key}_negated_mean_abs_error"] = float(
                    (rotated_component + base_component).abs().mean()
                )

    details["input_largest_drift_feature"] = largest_feature
    details["input_largest_drift_component"] = largest_component
    details["input_largest_drift_max_abs_difference"] = largest_maximum
    return details


def _top_disagreement_rows(
    *,
    base_probabilities: torch.Tensor,
    rotated_probabilities: torch.Tensor,
    data,
    common: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    difference = (rotated_probabilities - base_probabilities).abs().max(dim=-1).values
    count = min(limit, int(difference.numel()))
    if count == 0:
        return []
    indices = difference.topk(k=count).indices
    base_margin = _top_two_margin(base_probabilities)
    rotated_margin = _top_two_margin(rotated_probabilities)
    rows = []
    for index in indices.tolist():
        row = {
            **common,
            "prediction_index": index,
            "base_class": int(base_probabilities[index].argmax()),
            "rotated_class": int(rotated_probabilities[index].argmax()),
            "class_flipped": bool(
                base_probabilities[index].argmax() != rotated_probabilities[index].argmax()
            ),
            "base_top2_margin": float(base_margin[index]),
            "rotated_top2_margin": float(rotated_margin[index]),
            "max_abs_probability_difference": float(difference[index]),
        }
        if hasattr(data, "face") and int(data.face.shape[1]) == int(difference.numel()):
            centroid = data.pos[data.face[:, index].long()].mean(dim=0)
            row.update(
                {
                    "prediction_level": "face",
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]),
                }
            )
        elif int(data.pos.shape[0]) == int(difference.numel()):
            point = data.pos[index]
            row.update(
                {
                    "prediction_level": "vertex",
                    "centroid_x": float(point[0]),
                    "centroid_y": float(point[1]),
                    "centroid_z": float(point[2]),
                }
            )
        else:
            row["prediction_level"] = "other"
        rows.append(row)
    return rows


def _float_values(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


def _frame_diagnostics(cfg):
    node_features = set(cfg.features.get("node_features") or [])
    if {"dental_frame_cartesian", "dental_frame_cylindrical"} & node_features:
        return "dental_pca", compute_dental_pca_frame_diagnostics
    if {
        "stable_canonical_frame_cartesian",
        "stable_canonical_frame_cylindrical",
    } & node_features:
        return "stable_pca", compute_stable_pca_frame_diagnostics
    raise ValueError(
        "PCA stability requires dental_frame_* or stable_canonical_frame_* node features."
    )


def _rows_for_checkpoint(
    cfg,
    checkpoint_path: Path,
    fold: int,
    num_rotations: int,
    seed: int,
    device: torch.device,
    frame_name: str,
    frame_diagnostics,
    top_disagreements: int,
    node_feature_slices: list[tuple[str, slice]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datamodule = hydra.utils.instantiate(cfg.data, seed=cfg.seed, current_fold=fold)
    datamodule.setup("test")
    model = _load_model(cfg, checkpoint_path, device)
    generator = torch.Generator(device="cpu").manual_seed(seed + fold)
    rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []

    for loader_batch in tqdm(datamodule.test_dataloader(), desc=f"fold_{fold}_batches"):
        for data in loader_batch.to_data_list():
            base_frame = frame_diagnostics(data)
            base_probabilities, base_inputs, base_outputs = _predict_with_diagnostics(
                model, data, device
            )
            mesh_id = _mesh_id(data)
            rotations_tq = tqdm(range(num_rotations), desc=f"rotations_{mesh_id}", leave=False)
            for rotation_idx in rotations_tq:
                matrix, axis, angle = _axis_angle_rotation(generator)
                rotated_data = data.clone()
                _apply_rigid_transform(rotated_data, matrix=matrix)
                rotated_frame = frame_diagnostics(rotated_data)
                (
                    rotated_probabilities,
                    rotated_inputs,
                    rotated_outputs,
                ) = _predict_with_diagnostics(model, rotated_data, device)
    
                if rotated_probabilities.shape != base_probabilities.shape:
                    raise ValueError(
                        f"Prediction shape changed for {mesh_id}: "
                        f"{tuple(base_probabilities.shape)} -> {tuple(rotated_probabilities.shape)}"
                    )
                probability_difference = (rotated_probabilities - base_probabilities).abs()
                label_flips = (
                    rotated_probabilities.argmax(dim=-1)
                    != base_probabilities.argmax(dim=-1)
                )
                expected_frame = matrix @ base_frame["frame"]
                frame_difference = rotated_frame["frame"] - expected_frame
                base_margin = _top_two_margin(base_probabilities)
                rotated_margin = _top_two_margin(rotated_probabilities)
                flipped_margins = base_margin[label_flips]
                x_difference = _difference_stats(
                    base_inputs.get("x"), rotated_inputs.get("x")
                )
                feature_differences = _node_feature_difference_diagnostics(
                    base_inputs.get("x"),
                    rotated_inputs.get("x"),
                    node_feature_slices,
                )
                edge_difference = _difference_stats(
                    base_inputs.get("edge_attr"), rotated_inputs.get("edge_attr")
                )
                pos_difference = _equivariant_difference_stats(
                    base_inputs.get("pos"), rotated_inputs.get("pos"), matrix
                )
                normal_difference = _equivariant_difference_stats(
                    base_inputs.get("normal"), rotated_inputs.get("normal"), matrix
                )
                embedding_difference = _difference_stats(
                    base_outputs.get("embedding"), rotated_outputs.get("embedding")
                )
                raw_prediction_difference = _difference_stats(
                    base_outputs.get("raw_pred"), rotated_outputs.get("raw_pred")
                )

                row = {
                        "fold": fold,
                        "checkpoint": str(checkpoint_path),
                        "mesh_id": mesh_id,
                        "rotation_index": rotation_idx,
                        "axis_x": float(axis[0]),
                        "axis_y": float(axis[1]),
                        "axis_z": float(axis[2]),
                        "angle_radians": angle,
                        "frame_convention": frame_name,
                        "frame_frobenius_error": float(torch.linalg.norm(frame_difference)),
                        "frame_max_abs_error": float(frame_difference.abs().max()),
                        "base_eigenvalues": _float_values(base_frame["eigenvalues"]),
                        "base_relative_eigengaps": _float_values(base_frame["relative_eigengaps"]),
                        "base_sign_margins": _float_values(base_frame["sign_margins"]),
                        "rotated_relative_eigengaps": _float_values(rotated_frame["relative_eigengaps"]),
                        "rotated_sign_margins": _float_values(rotated_frame["sign_margins"]),
                        "prediction_items": int(base_probabilities.shape[0]),
                        "prediction_flip_count": int(label_flips.sum()),
                        "prediction_flip_rate": float(label_flips.float().mean()),
                        "prediction_mean_abs_probability_difference": float(probability_difference.mean()),
                        "prediction_max_abs_probability_difference": float(probability_difference.max()),
                        "base_top2_margin_mean": float(base_margin.mean()),
                        "flipped_base_top2_margin_mean": float(
                            flipped_margins.mean()
                            if flipped_margins.numel()
                            else 0.0
                        ),
                        "flipped_base_top2_margin_max": float(
                            flipped_margins.max()
                            if flipped_margins.numel()
                            else 0.0
                        ),
                    "input_x_mean_abs_difference": x_difference["mean"],
                    "input_x_max_abs_difference": x_difference["max"],
                    "input_feature_tensor_device": str(device),
                        "input_edge_attr_mean_abs_difference": edge_difference["mean"],
                        "input_edge_attr_max_abs_difference": edge_difference["max"],
                        "input_pos_equivariance_mean_abs_error": pos_difference["mean"],
                        "input_pos_equivariance_max_abs_error": pos_difference["max"],
                        "input_normal_equivariance_mean_abs_error": normal_difference["mean"],
                        "input_normal_equivariance_max_abs_error": normal_difference["max"],
                        "embedding_mean_abs_difference": embedding_difference["mean"],
                        "embedding_max_abs_difference": embedding_difference["max"],
                        "raw_prediction_mean_abs_difference": raw_prediction_difference["mean"],
                    "raw_prediction_max_abs_difference": raw_prediction_difference["max"],
                    **feature_differences,
                }
                rows.append(row)
                disagreement_rows.extend(
                    _top_disagreement_rows(
                        base_probabilities=base_probabilities,
                        rotated_probabilities=rotated_probabilities,
                        data=data,
                        common={
                            "fold": fold,
                            "mesh_id": mesh_id,
                            "rotation_index": rotation_idx,
                        },
                        limit=top_disagreements,
                    )
                )
            rotations_tq.close()
    return rows, disagreement_rows


def _write_results(
    rows: list[dict[str, Any]],
    disagreement_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pca_stability_rows.csv"
    json_path = output_dir / "pca_stability_summary.json"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    disagreement_path = output_dir / "pca_stability_top_disagreements.csv"
    if disagreement_rows:
        with disagreement_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(disagreement_rows[0]))
            writer.writeheader()
            writer.writerows(disagreement_rows)

    top_disagreements = sorted(
        rows,
        key=lambda row: (
            row["prediction_flip_rate"],
            row["prediction_max_abs_probability_difference"],
            row["prediction_mean_abs_probability_difference"],
        ),
        reverse=True,
    )[:20]
    mesh_summary: dict[str, dict[str, float]] = defaultdict(
        lambda: {"max_flip_rate": 0.0, "max_probability_difference": 0.0}
    )
    for row in rows:
        summary = mesh_summary[row["mesh_id"]]
        summary["max_flip_rate"] = max(summary["max_flip_rate"], row["prediction_flip_rate"])
        summary["max_probability_difference"] = max(
            summary["max_probability_difference"],
            row["prediction_max_abs_probability_difference"],
        )
    summary = {
        "num_rows": len(rows),
        "max_frame_frobenius_error": max(
            (row["frame_frobenius_error"] for row in rows), default=0.0
        ),
        "max_prediction_flip_rate": max(
            (row["prediction_flip_rate"] for row in rows), default=0.0
        ),
        "max_prediction_probability_difference": max(
            (row["prediction_max_abs_probability_difference"] for row in rows),
            default=0.0,
        ),
        "largest_prediction_disagreements": top_disagreements,
        "per_mesh_maximums": dict(sorted(mesh_summary.items())),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    if disagreement_rows:
        print(f"Wrote {disagreement_path}")
    print(f"Wrote {json_path}")
    _write_sensitivity_summary(rows, output_dir)


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (float, int)) and math.isfinite(value):
            values.append(float(value))
    return sorted(values)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = _finite_values(rows, key)
    if not values:
        return {}
    return {
        "median": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
        "mean": sum(values) / len(values),
    }


def _closest_quantile_row(
    rows: list[dict[str, Any]],
    key: str,
    quantile: float,
    used_meshes: set[str],
) -> dict[str, Any] | None:
    values = _finite_values(rows, key)
    if not values:
        return None
    target = _quantile(values, quantile)
    candidates = [row for row in rows if row["mesh_id"] not in used_meshes]
    if not candidates:
        return None
    # A quantile example should illustrate a substantial but non-extremal case.
    # Keep the global maximum for the full CSV rather than presenting it here.
    maximum = max(float(row.get(key, float("-inf"))) for row in candidates)
    non_maximum_candidates = [
        row for row in candidates if float(row.get(key, float("-inf"))) < maximum
    ]
    if non_maximum_candidates:
        candidates = non_maximum_candidates
    return min(
        candidates,
        key=lambda row: (abs(float(row.get(key, 0.0)) - target), row["rotation_index"]),
    )


def _stable_reference_row(
    rows: list[dict[str, Any]],
    used_meshes: set[str],
) -> dict[str, Any] | None:
    metric_keys = (
        "frame_frobenius_error",
        "input_largest_drift_max_abs_difference",
        "prediction_max_abs_probability_difference",
    )
    scales = {
        key: max(_quantile(_finite_values(rows, key), 0.9), 1e-12)
        for key in metric_keys
    }
    candidates = [row for row in rows if row["mesh_id"] not in used_meshes]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: sum(float(row.get(key, 0.0)) / scales[key] for key in metric_keys),
    )


def _representative_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    if not rows:
        return []
    used_meshes: set[str] = set()
    selected: list[tuple[str, dict[str, Any]]] = []
    selection_plan = [
        ("Stable reference", _stable_reference_row, None),
        ("Frame-disagreement example (90th percentile)", _closest_quantile_row, "frame_frobenius_error"),
    ]
    frame_convention = str(rows[0].get("frame_convention", ""))
    sensitivity_metric = (
        "input_largest_drift_max_abs_difference"
        if frame_convention == "dental_pca"
        else "prediction_max_abs_probability_difference"
    )
    sensitivity_label = (
        "Device-feature sensitivity example (90th percentile)"
        if frame_convention == "dental_pca"
        else "Prediction-sensitivity example (90th percentile)"
    )
    selection_plan.append((sensitivity_label, _closest_quantile_row, sensitivity_metric))

    for label, selector, metric_key in selection_plan:
        row = (
            selector(rows, used_meshes)
            if metric_key is None
            else selector(rows, metric_key, 0.9, used_meshes)
        )
        if row is None:
            continue
        selected.append((label, row))
        used_meshes.add(row["mesh_id"])
    return selected


def _format_float(value: Any) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        return "n/a"
    return f"{float(value):.3g}"


def _write_sensitivity_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    convention = str(rows[0]["frame_convention"])
    meshes = sorted({str(row["mesh_id"]) for row in rows})
    distributions = {
        key: _distribution(rows, key)
        for key in (
            "frame_frobenius_error",
            "input_largest_drift_max_abs_difference",
            "prediction_flip_rate",
            "prediction_max_abs_probability_difference",
        )
    }
    examples = _representative_rows(rows)
    summary = {
        "frame_convention": convention,
        "num_meshes": len(meshes),
        "num_rotated_evaluations": len(rows),
        "distributions": distributions,
        "representative_examples": [
            {"role": label, **row} for label, row in examples
        ],
    }
    json_path = output_dir / "pca_stability_sensitivity_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"# PCA sensitivity summary: {convention}",
        "",
        f"{len(meshes)} meshes; {len(rows)} mesh-rotation evaluations.",
        "",
        "## Distribution across mesh-rotation evaluations",
        "",
        "| Quantity | Median | 90th percentile | Mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("PCA frame Frobenius error", "frame_frobenius_error"),
        ("Largest device feature drift", "input_largest_drift_max_abs_difference"),
        ("Prediction flip rate", "prediction_flip_rate"),
        ("Maximum probability drift", "prediction_max_abs_probability_difference"),
    ):
        stats = distributions[key]
        lines.append(
            f"| {label} | {_format_float(stats.get('median'))} | "
            f"{_format_float(stats.get('p90'))} | {_format_float(stats.get('mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Representative distinct meshes",
            "",
            "These are quantile-near examples, deliberately not the global maxima.",
            "",
            "| Role | Mesh | Rotation | Frame error | Largest input drift | Source | Flip rate | Probability drift |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for label, row in examples:
        source = str(row.get("input_largest_drift_feature", ""))
        component = str(row.get("input_largest_drift_component", ""))
        if component and component != source:
            source = f"{source}/{component}"
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    str(row["mesh_id"]),
                    str(row["rotation_index"]),
                    _format_float(row["frame_frobenius_error"]),
                    _format_float(row["input_largest_drift_max_abs_difference"]),
                    source or "n/a",
                    _format_float(row["prediction_flip_rate"]),
                    _format_float(row["prediction_max_abs_probability_difference"]),
                )
            )
            + " |"
        )

    if convention == "dental_pca":
        lines.extend(
            [
                "",
                "For dental PCA, inspect the selected row's "
                "`input_feature_dental_frame_cylindrical_z_*` columns. "
                "High sign disagreement with low negated-value error indicates a Z-sign flip.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "For stable PCA, compare the stable-cylindrical component drift with frame error; "
                "large errors should concentrate around low-eigengap or low-sign-margin meshes.",
            ]
        )

    markdown_path = output_dir / "pca_stability_sensitivity_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    frame_p90 = distributions["frame_frobenius_error"].get("p90")
    feature_p90 = distributions["input_largest_drift_max_abs_difference"].get("p90")
    prediction_p90 = distributions[
        "prediction_max_abs_probability_difference"
    ].get("p90")
    print(
        "Sensitivity summary: "
        f"frame p90={_format_float(frame_p90)}, "
        f"feature-drift p90={_format_float(feature_p90)}, "
        f"probability-drift p90={_format_float(prediction_p90)}"
    )
    for label, row in examples:
        print(
            f"  {label}: mesh={row['mesh_id']} rotation={row['rotation_index']} "
            f"feature={row.get('input_largest_drift_feature', 'n/a')} "
            f"component={row.get('input_largest_drift_component', 'n/a')}"
        )
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")


def main() -> None:
    args = _parse_args()
    if args.num_rotations <= 0:
        raise ValueError("--num-rotations must be positive.")
    if args.top_disagreements <= 0:
        raise ValueError("--top-disagreements must be positive.")
    register_custom_omegaconf_resolvers()
    cfg, artifacts, _ = load_test_cfg_from_experiment_dir(str(args.experiment))
    cfg = _sync_runtime_config(cfg)
    data_target = str(cfg.data._target_).lower()
    if "liver" not in data_target and "teeth" not in data_target:
        raise ValueError(
            "This diagnostic supports LiverMeshDataModule and TeethMeshDataModule experiments."
        )
    device = _device_from_arg(args.device)
    frame_name, frame_diagnostics = _frame_diagnostics(cfg)
    node_feature_slices = _node_feature_slices(cfg)
    rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    for fold, checkpoint_path in _best_checkpoints(artifacts["checkpoints_dir"], args.fold):
        print(f"Evaluating fold {fold}: {checkpoint_path}")
        checkpoint_rows, checkpoint_disagreements = _rows_for_checkpoint(
            cfg,
            checkpoint_path,
            fold,
            args.num_rotations,
            args.seed,
            device,
            frame_name,
            frame_diagnostics,
            args.top_disagreements,
            node_feature_slices,
        )
        rows.extend(checkpoint_rows)
        disagreement_rows.extend(checkpoint_disagreements)
    output_dir = args.output_dir or artifacts["results_dir"] / "pca_stability"
    _write_results(rows, disagreement_rows, output_dir)


if __name__ == "__main__":
    main()
