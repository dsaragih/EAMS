from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data

from models.equiv_mesh_gnn_liver import EquivMeshGNNLiverClassifier
from util.config import register_custom_omegaconf_resolvers
from util.losses import EdgeLocalContinuityLoss


PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def _register_resolvers_once() -> None:
    if not OmegaConf.has_resolver("resolve_emnn_feature_input_dim"):
        register_custom_omegaconf_resolvers()


def test_edge_local_continuity_loss_penalizes_adjacent_disagreement():
    loss = EdgeLocalContinuityLoss(distance="tv")
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor([0, 0, -1], dtype=torch.long)

    aligned_logits = torch.tensor(
        [
            [6.0, -6.0, -6.0],
            [6.0, -6.0, -6.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    disagreeing_logits = torch.tensor(
        [
            [6.0, -6.0, -6.0],
            [-6.0, 6.0, -6.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    aligned_loss = loss(aligned_logits, targets, edge_index=edge_index)
    disagreeing_loss = loss(disagreeing_logits, targets, edge_index=edge_index)

    assert torch.isclose(aligned_loss, torch.tensor(0.0), atol=1e-4)
    assert disagreeing_loss > aligned_loss


def test_liver_continuity_loss_setup_hydra_compose():
    _register_resolvers_once()

    config_dir = PROJECT_ROOT / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=emnn_liver",
                "model/loss_setup=liver_vbdou_contrastive_continuity",
            ],
        )

    assert cfg.model.loss_setup.loss_dict.continuity._target_ == "util.losses.EdgeLocalContinuityLoss"
    assert cfg.model.loss_setup.loss_coefficients.continuity == 5.0
    # Paper bundle liver_local_continuity explicitly overrides the retained default to 1.0.


def test_liver_collect_loss_inputs_reuses_pred_for_continuity():
    model = EquivMeshGNNLiverClassifier.__new__(EquivMeshGNNLiverClassifier)
    model.losses = {
        "pred": object(),
        "continuity": EdgeLocalContinuityLoss(),
    }

    batch = Data(edge_labels=torch.tensor([0, 1], dtype=torch.long))
    pred_logits = torch.tensor(
        [[1.0, -1.0, 0.5], [0.2, 0.3, -0.4]], dtype=torch.float32
    )

    out_for_loss, labels = model._collect_loss_inputs(batch, {"pred": pred_logits})

    assert torch.equal(out_for_loss["pred"], pred_logits)
    assert torch.equal(out_for_loss["continuity"], pred_logits)
    assert torch.equal(labels["pred"], batch.edge_labels)