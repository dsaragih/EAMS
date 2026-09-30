import torch

from ..contracts import EncoderOutput


FACE_REDUCTIONS = {"first", "mean", "max", "sum"}


def derive_face_batch(
    face: torch.Tensor | None,
    batch: torch.Tensor,
) -> torch.Tensor | None:
    if face is None:
        return None
    if face.numel() == 0:
        return batch.new_empty((0,), dtype=batch.dtype)
    return batch[face[0].long()]


def pool_node_embedding_to_faces(
    node_embedding: torch.Tensor,
    face: torch.Tensor,
    reduce: str,
) -> torch.Tensor:
    if face.numel() == 0:
        return node_embedding.new_empty((0, node_embedding.shape[-1]))
    if reduce not in FACE_REDUCTIONS:
        raise ValueError(
            f"face reduction must be one of {sorted(FACE_REDUCTIONS)}, got: {reduce}"
        )

    gathered = node_embedding[face.long()]
    if reduce == "first":
        return gathered[0].contiguous()
    if reduce == "mean":
        return gathered.mean(dim=0)
    if reduce == "max":
        return gathered.max(dim=0).values
    return gathered.sum(dim=0)


def resolve_face_features(
    encoded: EncoderOutput,
    *,
    face_source: str = "auto",
    face_reduce: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if face_source not in {"auto", "face_embedding", "node_embedding"}:
        raise ValueError(
            "face_source must be one of {'auto', 'face_embedding', 'node_embedding'}"
        )

    if face_source in {"auto", "face_embedding"} and encoded.face_embedding is not None:
        face_batch = encoded.face_batch
        if face_batch is None:
            face_batch = derive_face_batch(encoded.face, encoded.batch)
        return encoded.face_embedding, face_batch

    if encoded.node_embedding is None:
        raise ValueError(
            "Face decoding requested node-derived features, but encoder output has no node_embedding"
        )
    if encoded.face is None:
        raise ValueError(
            "Face decoding requested node-derived features, but encoder output has no face connectivity"
        )

    face_features = pool_node_embedding_to_faces(
        node_embedding=encoded.node_embedding,
        face=encoded.face,
        reduce=face_reduce,
    )
    face_batch = encoded.face_batch
    if face_batch is None:
        face_batch = derive_face_batch(encoded.face, encoded.batch)
    return face_features, face_batch
