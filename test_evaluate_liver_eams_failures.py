from pathlib import Path

import pytest
import torch

from data.types import MeshData
from evaluate_liver_eams_failures import (
    UndirectedEdgeCollapseError,
    analyze_saved_mesh,
    collapse_undirected_edges,
    correlation_rows,
    correlation_summary,
    mean_logits_over_undirected_edges,
)


def _mesh_result() -> MeshData:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    face = torch.tensor([[0, 0, 0, 1], [2, 1, 3, 2], [1, 3, 2, 3]], dtype=torch.long)
    undirected = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 3, 3, 3]], dtype=torch.long)
    edge_index = torch.cat((undirected, undirected.flip(0)), dim=1)
    edge_labels = torch.tensor([0, 1, 2, 0, 1, 2] * 2, dtype=torch.long)
    predicted = torch.tensor([0, 1, 0, 0, 1, 2] * 2, dtype=torch.long)
    logits = torch.full((edge_labels.numel(), 3), -3.0)
    logits[torch.arange(edge_labels.numel()), predicted] = 3.0
    face_area = torch.full((face.shape[1],), 0.5)
    # face_adjacency indices are only needed by the dihedral helper.  The map
    # points each adjacent-face physical edge to its first directed edge.
    return MeshData(
        pos=pos,
        normal=torch.nn.functional.normalize(pos + 0.1, dim=-1),
        face=face,
        face_area=face_area,
        edge_index=edge_index,
        edge_labels=edge_labels,
        pred_edge_labels=predicted,
        pred_edge_logits=logits,
        face_adjacency=torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
        face_adjacency_edge_index=torch.tensor([0, 2, 6, 4, 8, 10]),
    )


def test_collapse_undirected_edges_preserves_physical_labels():
    data = _mesh_result()
    edges, labels = collapse_undirected_edges(data.edge_index, data.edge_labels)
    assert edges.shape == (2, 6)
    assert labels.tolist() == [0, 2, 0, 1, 1, 2]


def test_collapse_undirected_edges_rejects_inconsistent_opposites():
    with pytest.raises(UndirectedEdgeCollapseError):
        collapse_undirected_edges(
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([1, 2]),
        )


def test_mean_logits_resolves_directionally_different_predictions():
    edges, labels = mean_logits_over_undirected_edges(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([[0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]),
    )
    assert edges.tolist() == [[0], [1]]
    assert labels.tolist() == [2]


def test_analyze_saved_mesh_reports_dice_and_expected_descriptors(tmp_path: Path):
    path = tmp_path / "synthetic.pt"
    torch.save(_mesh_result(), path)
    row = analyze_saved_mesh(path, "experiment", 0)
    assert row["mesh_id"] == "synthetic"
    assert row["feature_num_vertices"] == 4
    assert row["feature_num_undirected_edges"] == 6
    assert row["ligament_dice"] == 1.0
    assert row["ridge_dice"] < 1.0
    assert 0.0 <= row["foreground_dice"] <= 1.0
    assert "feature_pca_eigengap_12" in row
    assert "feature_ligament_curvature_contrast" in row


def test_correlations_are_within_fold_and_summary_requires_consistency():
    rows = []
    for fold in (0, 1):
        for index in range(5):
            rows.append(
                {
                    "experiment_id": "experiment",
                    "fold": fold,
                    "foreground_dice": index / 4,
                    "ligament_dice": index / 4,
                    "ridge_dice": index / 4,
                    "feature_monotonic": index,
                    "feature_constant": 1.0,
                    "feature_prediction_margin_mean": index,
                    "feature_pca_eigenvalue_1": index,
                }
            )
    by_fold = correlation_rows(rows)
    monotonic = [row for row in by_fold if row["descriptor"] == "feature_monotonic" and row["outcome"] == "foreground_dice"]
    assert len(monotonic) == 2
    assert all(row["spearman_rho"] == pytest.approx(1.0) for row in monotonic)
    assert not any(row["descriptor"] == "feature_prediction_margin_mean" for row in by_fold)
    assert not any(row["descriptor"] == "feature_pca_eigenvalue_1" for row in by_fold)
    summary = correlation_summary(by_fold)
    item = next(row for row in summary if row["descriptor"] == "feature_monotonic" and row["outcome"] == "foreground_dice")
    assert item["consistent_association"] is True
