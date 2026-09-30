from typing import Any

import torch


def frame_output_count(sample: Any, level: str) -> int:
    if level == "vertex":
        return int(sample.pos.shape[0])
    if level == "edge":
        return int(sample.edge_index.shape[1])
    if level == "face":
        if getattr(sample, "face", None) is None:
            return 0
        return int(sample.face.shape[1])
    if level == "graph":
        return 1
    raise ValueError(f"Unsupported frame-averaging output level {level!r}.")


def feature_tensor_level(tensor_name: str) -> str:
    if tensor_name == "x":
        return "vertex"
    if tensor_name == "edge_attr":
        return "edge"
    if tensor_name == "face_attr":
        return "face"
    raise ValueError(f"Unsupported feature tensor name {tensor_name!r}.")


def repeat_feature_tensor_by_graph(
    tensor: torch.Tensor,
    samples: list[Any],
    *,
    tensor_name: str,
    repeats: int,
) -> torch.Tensor:
    level = feature_tensor_level(tensor_name)
    repeated = []
    offset = 0
    for sample in samples:
        count = frame_output_count(sample, level)
        chunk = tensor[offset : offset + count]
        offset += count
        for _ in range(repeats):
            repeated.append(chunk.clone())
    if not repeated:
        return tensor.new_empty((0, *tensor.shape[1:]))
    return torch.cat(repeated, dim=0)


def reuse_rigid_transform_invariant_features(
    *,
    source_batch: Any,
    target_batch: Any,
    samples: list[Any],
    repeats: int,
    featuriser: Any,
) -> bool:
    if featuriser is None:
        return False

    supports_reuse = getattr(
        featuriser, "supports_rigid_transform_feature_reuse", None
    )
    if not callable(supports_reuse) or not supports_reuse():
        return False

    invariant_tensors = featuriser.rigid_transform_invariant_tensors()
    reused_any = False
    for tensor_name in ("x", "edge_attr", "face_attr"):
        if not invariant_tensors.get(tensor_name, False):
            continue
        value = getattr(source_batch, tensor_name, None)
        if value is None:
            continue
        repeated = repeat_feature_tensor_by_graph(
            value,
            samples,
            tensor_name=tensor_name,
            repeats=repeats,
        )
        setattr(target_batch, tensor_name, repeated)
        reused_any = True
    return reused_any


def propagate_dynamic_node_attrs(batch: Any, samples: list[Any]) -> None:
    if not samples:
        return

    protected_attrs = {
        "pos",
        "x",
        "edge_index",
        "edge_attr",
        "face",
        "face_attr",
        "batch",
        "ptr",
    }
    node_counts = [int(sample.num_nodes) for sample in samples]

    for attr_name in batch.keys():
        if attr_name in protected_attrs:
            continue
        if all(hasattr(sample, attr_name) for sample in samples):
            continue

        value = getattr(batch, attr_name, None)
        if not torch.is_tensor(value) or value.ndim == 0:
            continue
        if int(value.shape[0]) != int(batch.num_nodes):
            continue

        offset = 0
        for sample, node_count in zip(samples, node_counts):
            sample_value = value[offset : offset + node_count]
            setattr(sample, attr_name, sample_value.clone())
            offset += node_count
