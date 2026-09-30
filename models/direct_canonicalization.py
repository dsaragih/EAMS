"""Direct SVD-projected, non-equivariant mesh canonicalization."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch_geometric.data import Batch

from data.transforms.transforms import _apply_rigid_transform


@dataclass
class DirectCanonicalizationResult:
    batch: Batch
    rotations: torch.Tensor
    raw_matrices: torch.Tensor
    singular_values: torch.Tensor
    projection_residuals: torch.Tensor


def near_identity_matrix(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a positive diagonal matrix with distinct singular values.

    Its SO(3) projection is identity, while its unequal diagonal entries avoid
    the repeated-singular-value SVD gradient ambiguity at exact identity.
    """
    return torch.tensor((1.0, 1.01, 1.02), device=device, dtype=dtype).diag()


def project_to_so3(raw_matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project batched 3-by-3 matrices to their closest proper rotations in FP32."""
    if raw_matrices.ndim != 3 or raw_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected raw matrices with shape (num_graphs, 3, 3), got "
            f"{tuple(raw_matrices.shape)}."
        )

    raw_fp32 = raw_matrices.float()
    with torch.autocast(device_type=raw_fp32.device.type, enabled=False):
        u, singular_values, vh = torch.linalg.svd(raw_fp32, full_matrices=False)
        uvh = u @ vh
        correction = torch.ones(
            (raw_fp32.shape[0], 3), device=raw_fp32.device, dtype=raw_fp32.dtype
        )
        correction[:, -1] = torch.where(
            torch.linalg.det(uvh) < 0,
            correction.new_tensor(-1.0),
            correction.new_tensor(1.0),
        )
        rotation = (u * correction.unsqueeze(-2)) @ vh
        residual = torch.linalg.matrix_norm(raw_fp32 - rotation, ord="fro", dim=(-2, -1))
    return rotation, singular_values, residual


def _canonical_pose(sample: Any, rotation: torch.Tensor, center: torch.Tensor) -> Any:
    """Center a row-vector mesh and express it in the predicted basis."""
    return _apply_rigid_transform(
        sample.clone(),
        matrix=rotation.transpose(0, 1),
        translation=-(center @ rotation),
    )


def direct_canonicalize_batch(
    *,
    batch: Batch,
    canonicalizer_net: Any,
    featuriser: Any,
    propagate_dynamic_node_attrs: Callable[[Any, list[Any]], None],
    reuse_features: Callable[..., bool],
) -> DirectCanonicalizationResult:
    """Predict one SO(3) basis per graph and rebuild the canonicalized batch."""
    if canonicalizer_net is None:
        raise ValueError("Direct canonicalization requires a canonicalizer network.")
    if featuriser is None:
        raise ValueError("Direct canonicalization requires a featuriser.")

    prediction = canonicalizer_net(batch)
    if prediction.get("level") != "graph":
        raise ValueError("Direct canonicalization requires graph-level matrix predictions.")
    residual = prediction["pred"]
    if residual.ndim != 2 or residual.shape[1] != 9:
        raise ValueError(
            "Direct canonicalization expected matrix residuals with shape "
            f"(num_graphs, 9), got {tuple(residual.shape)}."
        )

    samples = batch.to_data_list()
    propagate_dynamic_node_attrs(batch, samples)
    if len(samples) != int(residual.shape[0]):
        raise ValueError(
            "Direct canonicalization expected one matrix per graph: "
            f"graphs={len(samples)}, matrices={residual.shape[0]}."
        )

    base = near_identity_matrix(device=residual.device, dtype=torch.float32)
    raw_matrices = base.unsqueeze(0) + residual.float().reshape(-1, 3, 3)
    rotations, singular_values, projection_residuals = project_to_so3(raw_matrices)
    centers = torch.stack([sample.pos.float().mean(dim=0) for sample in samples])
    canonical_samples = [
        _canonical_pose(sample, rotations[idx], centers[idx])
        for idx, sample in enumerate(samples)
    ]
    canonical_batch = Batch.from_data_list(canonical_samples)
    if not reuse_features(
        source_batch=batch,
        target_batch=canonical_batch,
        samples=samples,
        repeats=1,
    ):
        canonical_batch = featuriser(canonical_batch)

    return DirectCanonicalizationResult(
        batch=canonical_batch,
        rotations=rotations,
        raw_matrices=raw_matrices,
        singular_values=singular_values,
        projection_residuals=projection_residuals,
    )
