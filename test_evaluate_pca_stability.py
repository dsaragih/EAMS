import torch
from torch_geometric.data import Data
from types import SimpleNamespace

from evaluate_pca_stability import (
    _frame_diagnostics,
    _node_feature_difference_diagnostics,
    _predict_probabilities,
    _representative_rows,
)


class _FaceSupervisedModel:
    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        return batch

    def __call__(self, batch):
        return {
            "pred": torch.tensor(
                [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]], device=batch.pos.device
            ),
            "level": "vertex",
        }

    def _get_face_logits(self, batch, output):
        return output["pred"][batch.face].mean(dim=0)


def test_teeth_predictions_are_compared_at_face_supervision_level():
    data = Data(
        pos=torch.zeros((3, 3)),
        face=torch.tensor([[0], [1], [2]], dtype=torch.long),
        face_labels=torch.tensor([0], dtype=torch.long),
    )

    probabilities = _predict_probabilities(
        _FaceSupervisedModel(), data, torch.device("cpu")
    )

    expected = torch.softmax(torch.tensor([[4.0 / 3.0, 2.0 / 3.0]]), dim=-1)
    assert torch.allclose(probabilities, expected)


def test_dental_frame_features_select_dental_pca_diagnostics():
    cfg = SimpleNamespace(features={"node_features": ["dental_frame_cylindrical"]})

    frame_name, diagnostics = _frame_diagnostics(cfg)

    assert frame_name == "dental_pca"
    assert diagnostics.__name__ == "compute_dental_pca_frame_diagnostics"


def test_feature_diagnostics_identify_a_cylindrical_z_sign_flip():
    base_x = torch.tensor([[1.0, 0.2, 0.1, 3.0], [1.0, 0.4, -0.2, -4.0]])
    rotated_x = base_x.clone()
    rotated_x[:, 3] *= -1.0

    diagnostics = _node_feature_difference_diagnostics(
        base_x,
        rotated_x,
        [("pointwise_area", slice(0, 1)), ("dental_frame_cylindrical", slice(1, 4))],
    )

    assert diagnostics["input_largest_drift_feature"] == "dental_frame_cylindrical"
    assert diagnostics["input_largest_drift_component"] == "z"
    assert (
        diagnostics[
            "input_feature_dental_frame_cylindrical_z_sign_disagreement_fraction"
        ]
        == 1.0
    )
    assert (
        diagnostics[
            "input_feature_dental_frame_cylindrical_z_negated_mean_abs_error"
        ]
        == 0.0
    )


def test_representative_summary_uses_distinct_quantile_near_meshes():
    rows = [
        {
            "mesh_id": f"mesh-{index}",
            "rotation_index": 0,
            "frame_convention": "dental_pca",
            "frame_frobenius_error": float(index),
            "input_largest_drift_max_abs_difference": float(index * 10),
            "prediction_max_abs_probability_difference": float(index) / 10.0,
        }
        for index in range(1, 6)
    ]

    examples = _representative_rows(rows)

    assert len(examples) == 3
    assert len({row["mesh_id"] for _, row in examples}) == 3
    assert examples[1][1]["mesh_id"] != "mesh-5"  # p90, not the maximum.
