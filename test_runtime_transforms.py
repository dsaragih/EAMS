import math
import random

import torch
from torch_geometric.data import Data

from data.transforms.transforms import (
    compose_graph_transforms,
    IgnoreClassLabels,
    _rotation_matrix,
    _stable_hash,
    RandomTCGNRotationMesh,
    ReflectMesh,
    RandomRotateMesh,
    RandomRotateTranslateMesh,
)


def _face_normal(data: Data, face_key: str = "face") -> torch.Tensor:
    face = getattr(data, face_key).long()
    v0 = data.pos[face[0, 0]]
    v1 = data.pos[face[1, 0]]
    v2 = data.pos[face[2, 0]]
    normal = torch.cross(v1 - v0, v2 - v0, dim=0)
    return torch.nn.functional.normalize(normal, dim=0, eps=1e-12)


def test_reflection_flips_face_winding_to_match_transformed_normals():
    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        face=torch.tensor([[0], [1], [2]], dtype=torch.long),
        face_1=torch.tensor([[0], [1], [2]], dtype=torch.long),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )

    transformed = ReflectMesh(axis="x")(data)

    expected_face = torch.tensor([[0], [2], [1]], dtype=torch.long)
    assert torch.equal(transformed.face, expected_face)
    assert torch.equal(transformed.face_1, expected_face)

    reflected_normal = transformed.normal[0]
    assert torch.allclose(_face_normal(transformed), reflected_normal, atol=1e-6)
    assert torch.allclose(
        _face_normal(transformed, "face_1"), reflected_normal, atol=1e-6
    )


def test_random_rotate_mesh_preserves_distances_and_normal_alignment():
    torch.manual_seed(0)

    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        face=torch.tensor([[0], [1], [2]], dtype=torch.long),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    original_dists = torch.cdist(data.pos, data.pos)

    transformed = RandomRotateMesh(degrees=(45.0, 45.0))(data)

    assert torch.allclose(
        torch.cdist(transformed.pos, transformed.pos), original_dists, atol=1e-5
    )
    assert torch.allclose(_face_normal(transformed), transformed.normal[0], atol=1e-6)


def test_tcgn_rotation_uses_one_shared_xyz_angle_and_rotates_normals():
    data = Data(
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        normal=torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
        mesh_id="shared-angle",
    )
    transform = RandomTCGNRotationMesh(degrees=27.0, seed=2024)
    matrix = transform.transform_matrix(data)
    transformed = transform(data.clone())

    # The shared-angle composition is the legacy TCGN X -> Y -> Z sequence.
    rng = random.Random(2024 + _stable_hash("shared-angle"))
    theta = rng.uniform(-27.0, 27.0)
    expected = (
        _rotation_matrix(theta, "z")
        @ _rotation_matrix(theta, "y")
        @ _rotation_matrix(theta, "x")
    )
    assert torch.allclose(matrix, expected, atol=1e-6)
    assert torch.allclose(transformed.pos, data.pos @ matrix.t(), atol=1e-6)
    assert torch.allclose(transformed.normal, data.normal @ matrix.t(), atol=1e-6)
    assert torch.allclose(matrix.t() @ matrix, torch.eye(3), atol=1e-6)


def test_zero_degree_tcgn_rotation_is_a_noop():
    data = Data(
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        normal=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )

    transformed = RandomTCGNRotationMesh(degrees=0.0)(data)

    assert transformed is data
    assert torch.equal(transformed.pos, data.pos)
    assert torch.equal(transformed.normal, data.normal)


def test_random_rotate_translate_mesh_applies_sampled_translation():
    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    original_pos = data.pos.clone()
    original_normal = data.normal.clone()
    expected_translation = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float32)

    transformed = RandomRotateTranslateMesh(
        degrees=(0.0, 0.0),
        x_bias=(1.5, 1.5),
        y_bias=(-2.0, -2.0),
        z_bias=(0.25, 0.25),
    )(data)

    assert torch.allclose(
        transformed.pos,
        torch.tensor(
            [
                [1.5, -2.0, 0.25],
                [2.5, 0.0, 3.25],
            ],
            dtype=torch.float32,
        ),
        atol=1e-6,
    )
    assert torch.allclose(transformed.normal, original_normal, atol=1e-6)
    assert torch.allclose(
        transformed.pos - original_pos,
        expected_translation.unsqueeze(0),
        atol=1e-6,
    )


def test_seeded_random_rotate_translate_mesh_runs_deterministically_at_test_stage():
    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        face=torch.tensor([[0], [1], [2]], dtype=torch.long),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        mesh_id="mesh-a",
    )

    transform_cfg = [
        {
            "_target_": "data.transforms.transforms.RandomRotateTranslateMesh",
            "degrees": 30.0,
            "x_bias": 0.2,
            "y_bias": 0.2,
            "z_bias": 0.2,
            "seed": 2024,
        }
    ]
    transform = compose_graph_transforms(transform_cfg, stage="test")

    out_a = transform(data.clone())
    out_b = transform(data.clone())

    assert torch.allclose(out_a.pos, out_b.pos, atol=1e-6)
    assert torch.allclose(out_a.normal, out_b.normal, atol=1e-6)
    assert not torch.allclose(out_a.pos, data.pos, atol=1e-6)


def test_seedless_random_rotate_translate_mesh_is_rejected_for_test_stage():
    transform_cfg = [
        {
            "_target_": "data.transforms.transforms.RandomRotateTranslateMesh",
            "degrees": 30.0,
            "x_bias": 0.2,
            "y_bias": 0.2,
            "z_bias": 0.2,
        }
    ]

    try:
        compose_graph_transforms(transform_cfg, stage="test")
    except ValueError as exc:
        assert "not deterministic" in str(exc)
    else:
        raise AssertionError("Expected non-deterministic test transform to be rejected")


def test_ignore_class_labels_masks_dental_molars_in_both_supervision_levels():
    data = Data(
        vertex_labels=torch.tensor([0, 8, 9, 16, -1], dtype=torch.long),
        face_labels=torch.tensor([8, 1, 16, -1], dtype=torch.long),
    )

    transformed = IgnoreClassLabels([8, 16])(data)

    assert torch.equal(
        transformed.vertex_labels, torch.tensor([0, -1, 9, -1, -1])
    )
    assert torch.equal(
        transformed.face_labels, torch.tensor([-1, 1, -1, -1])
    )
