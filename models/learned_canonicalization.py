"""Differentiable, finite-step SO(3) canonicalization for mesh batches."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch_geometric.data import Batch

from data.transforms.transforms import _apply_rigid_transform


@dataclass
class CanonicalizationResult:
    batch: Batch
    rotations: torch.Tensor
    seed_energies: torch.Tensor
    energy_history: torch.Tensor
    selected_seed: torch.Tensor
    eigengaps: torch.Tensor

    @property
    def energy_gap(self) -> torch.Tensor:
        if self.seed_energies.shape[1] < 2:
            return self.seed_energies.new_zeros((self.seed_energies.shape[0],))
        sorted_energies = self.seed_energies.sort(dim=1).values
        return sorted_energies[:, 1] - sorted_energies[:, 0]


def _hat(vectors: torch.Tensor) -> torch.Tensor:
    x, y, z = vectors.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def right_handed_pca_seeds(sample: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four det=+1 PCA sign choices, their centroid, and eigengaps."""
    pos = sample.pos.float()
    center = pos.mean(dim=0) if pos.numel() else pos.new_zeros(3)
    centered = pos - center
    if centered.shape[0] < 3:
        basis = torch.eye(3, device=pos.device, dtype=pos.dtype)
        eigengaps = pos.new_zeros(2)
    else:
        covariance = centered.transpose(0, 1).matmul(centered) / float(centered.shape[0])
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.arange(2, -1, -1, device=pos.device)
        eigenvalues = eigenvalues[order]
        basis = eigenvectors[:, order]
        scale = eigenvalues[0].abs().clamp_min(1.0e-12)
        eigengaps = torch.stack(
            [(eigenvalues[0] - eigenvalues[1]) / scale, (eigenvalues[1] - eigenvalues[2]) / scale]
        )

    signs = pos.new_tensor(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    )
    return basis.unsqueeze(0) * signs.unsqueeze(1), center, eigengaps


def _apply_canonical_pose(sample: Any, rotation: torch.Tensor, center: torch.Tensor) -> Any:
    transformed = sample.clone()
    return _apply_rigid_transform(
        transformed,
        matrix=rotation.transpose(0, 1),
        translation=-(center @ rotation),
    )


def _build_candidate_batch(
    *,
    samples: list[Any],
    rotations: torch.Tensor,
    centers: torch.Tensor,
    source_batch: Batch,
    featuriser: Any,
    reuse_features: Callable[..., bool],
) -> Batch:
    candidate_samples = []
    frame_count = int(rotations.shape[1])
    for graph_idx, sample in enumerate(samples):
        for frame_idx in range(frame_count):
            candidate_samples.append(
                _apply_canonical_pose(sample, rotations[graph_idx, frame_idx], centers[graph_idx])
            )
    candidate_batch = Batch.from_data_list(candidate_samples)
    if not reuse_features(
        source_batch=source_batch,
        target_batch=candidate_batch,
        samples=samples,
        repeats=frame_count,
    ):
        candidate_batch = featuriser(candidate_batch)
    return candidate_batch


def _energy_values(energy_net: Any, candidate_batch: Batch, *, num_graphs: int, frame_count: int) -> torch.Tensor:
    output = energy_net(candidate_batch)
    if output.get("level") != "graph":
        raise ValueError("Learned canonicalization requires graph-level scalar energies.")
    values = output["pred"]
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.squeeze(1)
    if values.ndim != 1 or int(values.shape[0]) != num_graphs * frame_count:
        raise ValueError(
            "Learned canonicalization expected one scalar energy per PCA seed."
        )
    return values.reshape(num_graphs, frame_count)


def _rotation_gradients(candidate_batch: Batch, position_grad: torch.Tensor, samples: list[Any], frame_count: int) -> torch.Tensor:
    gradients = []
    offset = 0
    for sample in samples:
        count = int(sample.pos.shape[0])
        for _ in range(frame_count):
            position = candidate_batch.pos[offset : offset + count]
            derivative = position_grad[offset : offset + count]
            gradients.append(torch.cross(derivative, position, dim=-1).sum(dim=0))
            offset += count
    return torch.stack(gradients).reshape(len(samples), frame_count, 3)


def _bounded_so3_update(
    tangent_gradient: torch.Tensor, *, step_size: float, max_step_angle: float
) -> torch.Tensor:
    update = -step_size * tangent_gradient
    update_norm = update.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return update * (max_step_angle / update_norm).clamp_max(1.0)


def _canonicalize_batch(
    *,
    batch: Batch,
    energy_net: Any,
    featuriser: Any,
    propagate_dynamic_node_attrs: Callable[[Any, list[Any]], None],
    reuse_features: Callable[..., bool],
    steps: int,
    step_size: float,
    differentiable: bool,
    max_step_angle: float = 0.25,
) -> CanonicalizationResult:
    if energy_net is None:
        raise ValueError("Learned canonicalization requires an energy network.")
    if featuriser is None:
        raise ValueError("Learned canonicalization requires a featuriser.")
    if steps < 1:
        raise ValueError("learned_canonicalization.steps must be positive.")
    if step_size <= 0.0:
        raise ValueError("learned_canonicalization.step_size must be positive.")
    if max_step_angle <= 0.0:
        raise ValueError("learned_canonicalization.max_step_angle must be positive.")

    samples = batch.to_data_list()
    propagate_dynamic_node_attrs(batch, samples)
    if not samples:
        raise ValueError("Learned canonicalization requires at least one graph.")

    seed_data = [right_handed_pca_seeds(sample) for sample in samples]
    rotations = torch.stack([item[0] for item in seed_data]).detach().requires_grad_(True)
    centers = torch.stack([item[1] for item in seed_data])
    eigengaps = torch.stack([item[2] for item in seed_data])
    frame_count = int(rotations.shape[1])
    energy_history = []
    autocast_device = batch.pos.device.type

    with torch.enable_grad(), torch.autocast(
        device_type=autocast_device, enabled=False
    ):
        for step in range(steps):
            candidate_batch = _build_candidate_batch(
                samples=samples,
                rotations=rotations,
                centers=centers,
                source_batch=batch,
                featuriser=featuriser,
                reuse_features=reuse_features,
            )
            energies = _energy_values(
                energy_net,
                candidate_batch,
                num_graphs=len(samples),
                frame_count=frame_count,
            )
            energy_history.append(energies.detach())
            position_grad = torch.autograd.grad(
                energies.sum(),
                candidate_batch.pos,
                create_graph=differentiable,
            )[0]
            tangent_gradient = _rotation_gradients(
                candidate_batch, position_grad, samples, frame_count
            )
            update = _bounded_so3_update(
                tangent_gradient,
                step_size=step_size,
                max_step_angle=max_step_angle,
            )
            rotations = rotations @ torch.linalg.matrix_exp(_hat(update))
            if step + 1 < steps:
                # Keep the final solver update differentiable without nesting
                # higher-order derivatives through every earlier update.
                rotations = rotations.detach().requires_grad_(True)

        candidate_batch = _build_candidate_batch(
            samples=samples,
            rotations=rotations,
            centers=centers,
            source_batch=batch,
            featuriser=featuriser,
            reuse_features=reuse_features,
        )
        final_energies = _energy_values(
            energy_net,
            candidate_batch,
            num_graphs=len(samples),
            frame_count=frame_count,
        )
        energy_history.append(final_energies.detach())

    selected_seed = final_energies.detach().argmin(dim=1)
    selected_rotations = rotations[
        torch.arange(len(samples), device=rotations.device), selected_seed
    ]
    canonical_samples = [
        _apply_canonical_pose(sample, selected_rotations[idx], centers[idx])
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

    return CanonicalizationResult(
        batch=canonical_batch,
        rotations=selected_rotations,
        seed_energies=final_energies.detach(),
        energy_history=torch.stack(energy_history, dim=1),
        selected_seed=selected_seed,
        eigengaps=eigengaps,
    )


def canonicalize_batch(**kwargs) -> CanonicalizationResult:
    """Run the solver with autograd available during Lightning inference."""
    with torch.inference_mode(False):
        return _canonicalize_batch(**kwargs)
