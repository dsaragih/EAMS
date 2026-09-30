import torch
from torch_geometric.data import Data

from data.features.node_features import (
    compute_dental_frame,
    compute_dental_pca_frame_diagnostics,
    compute_stable_pca_frame_diagnostics,
)
from data.transforms.transforms import RotateMesh


def test_stable_pca_diagnostics_frame_is_rotation_covariant_when_unambiguous():
    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.1, 0.0],
                [0.2, 1.1, 0.0],
                [0.1, 0.3, 0.7],
                [1.5, 0.8, 0.2],
            ],
            dtype=torch.float32,
        ),
        face=torch.tensor(
            [[0, 0, 0, 1], [1, 2, 3, 2], [2, 3, 4, 4]], dtype=torch.long
        ),
    )
    base = compute_stable_pca_frame_diagnostics(data)
    transform = RotateMesh(37.0, axis="y")
    rotated = compute_stable_pca_frame_diagnostics(transform(data.clone()))

    assert torch.allclose(
        rotated["frame"], transform.transform_matrix() @ base["frame"], atol=1e-5
    )
    assert torch.all(base["relative_eigengaps"] > 0.05)


def test_dental_pca_diagnostics_match_dental_frame_coordinates():
    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.1, 0.0],
                [0.2, 1.1, 0.0],
                [0.1, 0.3, 0.7],
                [1.5, 0.8, 0.2],
            ],
            dtype=torch.float32,
        )
    )
    diagnostics = compute_dental_pca_frame_diagnostics(data)
    local = (data.pos - diagnostics["centroid"]) @ diagnostics["frame"]
    local[:, 1] -= local[:, 1].min()

    assert torch.allclose(
        local, compute_dental_frame(data, coordinates="cartesian"), atol=1e-6
    )

    transform = RotateMesh(37.0, axis="y")
    rotated = compute_dental_pca_frame_diagnostics(transform(data.clone()))
    assert torch.allclose(
        rotated["frame"], transform.transform_matrix() @ diagnostics["frame"], atol=1e-5
    )
