from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.data import Batch

from data.transforms.transforms import _apply_rigid_transform

from .networks.contracts import EncoderOutput
from .rigid_batch_utils import frame_output_count


@dataclass
class FrameAveragingBatch:
    framed_batch: Batch
    samples: list[Any]
    frame_count: int
    frame_rotations: torch.Tensor


@dataclass
class FrameAveragingSettings:
    """Validated access to the existing Hydra frame-averaging mapping."""

    config: Any

    def get(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            return default
        return self.config.get(key, default)

    def nested_get(self, section: str, key: str, default: Any = None) -> Any:
        return self.get(section, {}).get(key, default)

    @staticmethod
    def positive(value: float, config_key: str) -> float:
        if value <= 0.0:
            raise ValueError(f"{config_key} must be positive, got {value!r}.")
        return value

    def schedule_progress(
        self, *, current_epoch: int, max_epochs: int | None, fallback_max_epochs: int
    ) -> float:
        if max_epochs is None or int(max_epochs) <= 0:
            max_epochs = fallback_max_epochs
        if int(max_epochs) <= 1:
            return 1.0
        return min(max(float(current_epoch) / float(max_epochs - 1), 0.0), 1.0)

    def tau(
        self, *, current_epoch: int, max_epochs: int | None, fallback_max_epochs: int
    ) -> float:
        base = self.positive(float(self.get("tau", 1.0)), "frame_averaging.tau")
        annealing = self.get("temperature_annealing", {})
        if not bool(annealing.get("enabled", False)):
            return base
        start = self.positive(
            float(annealing.get("initial_tau", base)),
            "frame_averaging.temperature_annealing.initial_tau",
        )
        end = self.positive(
            float(annealing.get("final_tau", base)),
            "frame_averaging.temperature_annealing.final_tau",
        )
        progress = self.schedule_progress(
            current_epoch=current_epoch,
            max_epochs=max_epochs,
            fallback_max_epochs=fallback_max_epochs,
        )
        return start + (end - start) * progress

    def energy_scheduler_value(self, key: str, default: Any = None) -> Any:
        return self.nested_get("energy_training", "scheduler", {}).get(key, default)



def frame_averaging_signs(
    *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    values = torch.tensor([-1.0, 1.0], device=device, dtype=dtype)
    grids = torch.meshgrid(values, values, values, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1)


def pca_frame_matrices(
    sample: Any, *, frame_group: str = "o3"
) -> tuple[torch.Tensor, torch.Tensor]:
    if frame_group not in {"o3", "so3"}:
        raise ValueError(
            "frame_group must be one of {'o3', 'so3'}, "
            f"got {frame_group!r}."
        )

    pos = sample.pos.float()
    center = pos.mean(dim=0)
    centered = pos - center
    if centered.shape[0] < 3 or centered.numel() == 0:
        basis = torch.eye(3, device=pos.device, dtype=pos.dtype)
    else:
        cov = centered.t().matmul(centered) / float(centered.shape[0])
        _, evecs = torch.linalg.eigh(cov)
        order = torch.arange(2, -1, -1, device=pos.device)
        basis = evecs[:, order]

    signs = frame_averaging_signs(device=pos.device, dtype=pos.dtype)
    frames = basis.unsqueeze(0) * signs.unsqueeze(1)
    if frame_group == "so3":
        frames = frames[torch.linalg.det(frames.float()) > 0.0]
    return frames, center


def sample_haar_rotation_matrices(
    *, count: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if count <= 0:
        return torch.empty((0, 3, 3), device=device, dtype=dtype)

    quaternions = torch.randn((count, 4), device=device, dtype=dtype)
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True).clamp_min(
        1.0e-12
    )
    q0, qx, qy, qz = quaternions.unbind(dim=1)

    rotations = torch.empty((count, 3, 3), device=device, dtype=dtype)
    rotations[:, 0, 0] = 1.0 - 2.0 * (qy.square() + qz.square())
    rotations[:, 0, 1] = 2.0 * (qx * qy - q0 * qz)
    rotations[:, 0, 2] = 2.0 * (qx * qz + q0 * qy)
    rotations[:, 1, 0] = 2.0 * (qx * qy + q0 * qz)
    rotations[:, 1, 1] = 1.0 - 2.0 * (qx.square() + qz.square())
    rotations[:, 1, 2] = 2.0 * (qy * qz - q0 * qx)
    rotations[:, 2, 0] = 2.0 * (qx * qz - q0 * qy)
    rotations[:, 2, 1] = 2.0 * (qy * qz + q0 * qx)
    rotations[:, 2, 2] = 1.0 - 2.0 * (qx.square() + qy.square())
    return rotations


def build_frame_matrices(
    sample: Any,
    *,
    include_pca_frames: bool,
    pca_frame_group: str = "o3",
    include_identity_frame: bool,
    num_random_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = sample.pos.float()
    center = pos.mean(dim=0)
    frames = []

    if include_pca_frames:
        pca_frames, center = pca_frame_matrices(sample, frame_group=pca_frame_group)
        frames.append(pca_frames)

    if include_identity_frame:
        identity = torch.eye(3, device=pos.device, dtype=pos.dtype).unsqueeze(0)
        frames.append(identity)

    if num_random_frames > 0:
        frames.append(
            sample_haar_rotation_matrices(
                count=num_random_frames,
                device=pos.device,
                dtype=pos.dtype,
            )
        )

    if not frames:
        raise ValueError(
            "Frame averaging requires at least one enabled frame source. "
            "Enable PCA frames, the identity frame, or random frames."
        )

    return torch.cat(frames, dim=0), center


def infer_frame_tensor_level(
    tensor: torch.Tensor, samples: list[Any], frame_count: int
) -> str | None:
    if tensor.ndim == 0:
        return None
    totals = {
        "vertex": sum(frame_output_count(sample, "vertex") for sample in samples)
        * frame_count,
        "edge": sum(frame_output_count(sample, "edge") for sample in samples)
        * frame_count,
        "face": sum(frame_output_count(sample, "face") for sample in samples)
        * frame_count,
        "graph": len(samples) * frame_count,
    }
    for level, total in totals.items():
        if int(tensor.shape[0]) == int(total):
            return level
    return None


def frame_chunk_tensor(
    tensor: torch.Tensor,
    samples: list[Any],
    *,
    frame_count: int,
    level: str,
) -> list[torch.Tensor]:
    chunked = []
    offset = 0
    for sample in samples:
        count = frame_output_count(sample, level)
        if count == 0:
            chunked.append(tensor.new_empty((frame_count, 0, *tensor.shape[1:])))
            continue

        frame_chunks = []
        for _ in range(frame_count):
            frame_chunks.append(tensor[offset : offset + count])
            offset += count
        chunked.append(torch.stack(frame_chunks, dim=0))
    return chunked


def average_frame_tensor(
    tensor: torch.Tensor,
    samples: list[Any],
    *,
    frame_count: int,
    level: str,
) -> torch.Tensor:
    averaged = []
    for frame_chunks in frame_chunk_tensor(
        tensor,
        samples,
        frame_count=frame_count,
        level=level,
    ):
        if frame_chunks.shape[1] == 0:
            averaged.append(tensor.new_empty((0, *tensor.shape[1:])))
            continue
        averaged.append(frame_chunks.mean(dim=0))
    if not averaged:
        return tensor.new_empty((0, *tensor.shape[1:]))
    return torch.cat(averaged, dim=0)


def weighted_frame_tensor(
    tensor: torch.Tensor,
    samples: list[Any],
    *,
    frame_count: int,
    level: str,
    frame_weights: torch.Tensor,
) -> torch.Tensor:
    weighted = []
    for graph_idx, frame_chunks in enumerate(
        frame_chunk_tensor(
            tensor,
            samples,
            frame_count=frame_count,
            level=level,
        )
    ):
        if frame_chunks.shape[1] == 0:
            weighted.append(tensor.new_empty((0, *tensor.shape[1:])))
            continue
        view_shape = (frame_count,) + (1,) * (frame_chunks.ndim - 1)
        weights = frame_weights[graph_idx].reshape(view_shape)
        weighted.append((frame_chunks * weights).sum(dim=0))
    if not weighted:
        return tensor.new_empty((0, *tensor.shape[1:]))
    return torch.cat(weighted, dim=0)


def select_frame_tensor(
    tensor: torch.Tensor,
    samples: list[Any],
    *,
    frame_count: int,
    level: str,
    selected_frame_indices: torch.Tensor,
) -> torch.Tensor:
    selected = []
    for graph_idx, frame_chunks in enumerate(
        frame_chunk_tensor(
            tensor,
            samples,
            frame_count=frame_count,
            level=level,
        )
    ):
        if frame_chunks.shape[1] == 0:
            selected.append(tensor.new_empty((0, *tensor.shape[1:])))
            continue
        selected.append(frame_chunks[int(selected_frame_indices[graph_idx].item())])
    if not selected:
        return tensor.new_empty((0, *tensor.shape[1:]))
    return torch.cat(selected, dim=0)


def build_frame_averaging_batch(
    batch: Batch,
    *,
    include_pca_frames: bool,
    pca_frame_group: str = "o3",
    include_identity_frame: bool,
    num_random_frames: int,
    featuriser: Any,
    propagate_dynamic_node_attrs: Any,
    reuse_rigid_transform_invariant_features: Any,
) -> FrameAveragingBatch:
    if featuriser is None:
        raise ValueError("Frame averaging requires a configured featuriser.")

    samples = batch.to_data_list()
    propagate_dynamic_node_attrs(batch, samples)
    frame_count = None
    framed_samples = []
    frame_rotations = []

    for sample in samples:
        frames, center = build_frame_matrices(
            sample,
            include_pca_frames=include_pca_frames,
            pca_frame_group=pca_frame_group,
            include_identity_frame=include_identity_frame,
            num_random_frames=num_random_frames,
        )
        if frame_count is None:
            frame_count = int(frames.shape[0])
        elif int(frames.shape[0]) != int(frame_count):
            raise ValueError(
                "Frame averaging requires a consistent frame count across graphs, "
                f"got {frame_count} and {frames.shape[0]}."
            )
        frame_rotations.append(frames)
        for frame in frames:
            framed = sample.clone()
            framed = _apply_rigid_transform(
                framed,
                matrix=frame.t(),
                translation=-(center @ frame),
            )
            framed_samples.append(framed)

    framed_batch = Batch.from_data_list(framed_samples)
    reused = reuse_rigid_transform_invariant_features(
        source_batch=batch,
        target_batch=framed_batch,
        samples=samples,
        repeats=frame_count,
    )
    if not reused:
        framed_batch = featuriser(framed_batch)

    if frame_count is None:
        frame_count = 0
    if frame_rotations:
        frame_rotations_tensor = torch.stack(frame_rotations, dim=0)
    else:
        dtype = batch.pos.dtype if hasattr(batch, "pos") else torch.float32
        device = batch.pos.device if hasattr(batch, "pos") else framed_batch.x.device
        frame_rotations_tensor = torch.empty(
            (0, frame_count, 3, 3), dtype=dtype, device=device
        )

    return FrameAveragingBatch(
        framed_batch=framed_batch,
        samples=samples,
        frame_count=frame_count,
        frame_rotations=frame_rotations_tensor,
    )


def uniform_frame_weights(
    *,
    num_graphs: int,
    frame_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_graphs == 0:
        return torch.empty((0, frame_count), device=device, dtype=dtype)
    return torch.full(
        (num_graphs, frame_count),
        fill_value=1.0 / float(frame_count),
        device=device,
        dtype=dtype,
    )


def prepare_frame_energy_batch(
    *,
    framed_batch: Batch,
    embedding: torch.Tensor | None,
) -> Batch:
    if embedding is None:
        return framed_batch
    if int(embedding.shape[0]) != int(framed_batch.num_nodes):
        raise ValueError(
            "Expanded embedding shape must align with framed_batch.num_nodes: "
            f"embedding has {embedding.shape[0]} rows but framed_batch has "
            f"{framed_batch.num_nodes} nodes."
        )

    energy_batch = framed_batch.clone()
    energy_batch.x = embedding
    return energy_batch


def sample_frame_indices(
    *,
    frame_weights: torch.Tensor,
    uses_learned_weights: bool,
    frame_count: int,
) -> torch.Tensor:
    if frame_weights.shape[0] == 0:
        return torch.empty((0,), device=frame_weights.device, dtype=torch.long)
    if uses_learned_weights:
        return torch.multinomial(frame_weights, num_samples=1).squeeze(1)
    return torch.randint(
        frame_count,
        (frame_weights.shape[0],),
        device=frame_weights.device,
    )


def frame_averaging_metadata(
    *,
    stage: str,
    frame_weights: torch.Tensor,
    frame_rotations: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metadata = {"frame_weights": frame_weights}
    if stage == "test":
        metadata["frame_rotations"] = frame_rotations
    return metadata


def collapse_non_equivariant_encoder_output(
    *,
    expanded_encoded: EncoderOutput,
    original_batch: Batch,
    samples: list[Any],
    frame_count: int,
    frame_weights: torch.Tensor,
    frame_rotations: torch.Tensor,
) -> EncoderOutput:
    if expanded_encoded.node_embedding is None:
        raise ValueError("Frame averaging encoder collapse requires node_embedding.")
    if expanded_encoded.latent_pos is None:
        raise ValueError("Frame averaging encoder collapse requires latent_pos.")
    if expanded_encoded.edge_index is None or expanded_encoded.edge_attr is None:
        raise ValueError(
            "Frame averaging encoder collapse requires edge_index and edge_attr."
        )

    node_embedding = weighted_frame_tensor(
        expanded_encoded.node_embedding,
        samples,
        frame_count=frame_count,
        level="vertex",
        frame_weights=frame_weights,
    )
    edge_attr = weighted_frame_tensor(
        expanded_encoded.edge_attr,
        samples,
        frame_count=frame_count,
        level="edge",
        frame_weights=frame_weights,
    )

    latent_chunks = frame_chunk_tensor(
        expanded_encoded.latent_pos,
        samples,
        frame_count=frame_count,
        level="vertex",
    )
    collapsed_latent = []
    for graph_idx, frame_chunks in enumerate(latent_chunks):
        if frame_chunks.shape[1] == 0:
            collapsed_latent.append(
                expanded_encoded.latent_pos.new_empty(
                    (0, *expanded_encoded.latent_pos.shape[1:])
                )
            )
            continue
        rotated_back = torch.matmul(frame_chunks, frame_rotations[graph_idx].transpose(1, 2))
        weights = frame_weights[graph_idx].view(frame_count, 1, 1)
        collapsed_latent.append((rotated_back * weights).sum(dim=0))
    latent_pos = torch.cat(collapsed_latent, dim=0)

    face = getattr(original_batch, "face", None)
    face_batch = getattr(original_batch, "face_batch", None)
    face_embedding = None
    if expanded_encoded.face_embedding is not None:
        if face is None:
            raise ValueError(
                "Frame averaging encoder collapse received face_embedding without face connectivity."
            )
        face_embedding = weighted_frame_tensor(
            expanded_encoded.face_embedding,
            samples,
            frame_count=frame_count,
            level="face",
            frame_weights=frame_weights,
        )
        if face_batch is None:
            if face.numel() > 0:
                face_batch = original_batch.batch[face[0].long()]
            else:
                face_batch = original_batch.batch.new_empty(
                    (0,), dtype=original_batch.batch.dtype
                )

    return EncoderOutput(
        node_embedding=node_embedding,
        pos=original_batch.pos,
        batch=original_batch.batch,
        latent_pos=latent_pos,
        edge_index=original_batch.edge_index,
        edge_attr=edge_attr,
        face=face,
        face_batch=face_batch,
        face_embedding=face_embedding,
    )
