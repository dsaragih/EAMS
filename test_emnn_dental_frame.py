from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data

from data.features.factory import FeatureFactory
from data.features.node_features import (
    compute_bbox_center_coordinates,
    compute_bbox_vertical_position,
    compute_canonical_frame,
    compute_com_fps_anchor_distances,
    compute_distance_to_bbox_mid_sagittal_plane,
    compute_gaussian_curvature,
    compute_oriented_bbox_frame,
    compute_principal_curvatures,
    compute_signed_bbox_midline_offset,
    compute_stable_canonical_frame,
    compute_thickness,
)
from data.features.edge_features import (
    compute_edge_midline_proximity,
    compute_edge_vertical_alignment,
    compute_signed_dihedrals,
    compute_chiral_edge_volume,
)
from models.networks.encoders.emnn import EMNN
from util.config import register_custom_omegaconf_resolvers


PROJECT_ROOT = Path(__file__).parent.absolute()


def _register_resolvers_once() -> None:
    if not OmegaConf.has_resolver("resolve_emnn_feature_input_dim"):
        register_custom_omegaconf_resolvers()


def test_emnn_dental_frame_hydra_resolvers():
    _register_resolvers_once()

    config_dir = PROJECT_ROOT / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg_with_frame = compose(
            config_name="config",
            overrides=["model=emnn_teeth", "features=vertex_hks_dihedrals_frame_cart"],
        )
        assert cfg_with_frame.model.encoder_config.input_dim == 9
        assert cfg_with_frame.model.encoder_config.num_vector == 3

        cfg_without_frame = compose(
            config_name="config",
            overrides=["model=emnn_teeth", "features=vertex_hks_dihedrals"],
        )
        assert cfg_without_frame.model.encoder_config.input_dim == 9
        assert cfg_without_frame.model.encoder_config.num_vector == 2

        cfg_frame_as_features = compose(
            config_name="config",
            overrides=[
                "model=emnn_teeth",
                "features=vertex_hks_dihedrals_frame_cart",
                "model.encoder_config.use_auxiliary_frame=false",
            ],
        )
        assert cfg_frame_as_features.model.encoder_config.input_dim == 12
        assert cfg_frame_as_features.model.encoder_config.num_vector == 2


def test_feature_factory_stores_dental_frame_separately():
    featuriser = FeatureFactory(
        node_features=["background_binary", "dental_frame_cartesian"],
        edge_features=[],
    )

    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        vertex_labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )

    out = featuriser(data)

    assert hasattr(out, "dental_frame")
    assert out.dental_frame.shape == (4, 3)
    assert out.x.shape == (4, 1)




def test_feature_factory_stores_canonical_frame_separately():
    featuriser = FeatureFactory(
        node_features=["background_binary", "canonical_frame_cartesian"],
        edge_features=[],
    )

    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        vertex_labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )

    out = featuriser(data)

    assert hasattr(out, "canonical_frame")
    assert out.canonical_frame.shape == (4, 3)
    assert out.x.shape == (4, 1)


def test_feature_factory_stores_stable_canonical_frame_separately():
    featuriser = FeatureFactory(
        node_features=["background_binary", "stable_canonical_frame_cartesian"],
        edge_features=[],
    )

    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.2],
            ],
            dtype=torch.float32,
        ),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        vertex_labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        face=torch.tensor(
            [
                [0, 0, 0, 1],
                [1, 2, 3, 2],
                [2, 3, 1, 3],
            ],
            dtype=torch.long,
        ),
        face_area=torch.tensor([0.75, 0.60, 0.90, 0.70], dtype=torch.float32),
    )

    out = featuriser(data)

    assert hasattr(out, "canonical_frame")
    assert out.canonical_frame.shape == (4, 3)
    assert out.x.shape == (4, 1)


def test_emnn_can_treat_canonical_frame_as_node_features():
    featuriser = FeatureFactory(
        node_features=["background_binary", "canonical_frame_cartesian"],
        edge_features=[],
    )

    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        vertex_labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        edge_index=torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
        ),
        face=torch.empty((3, 0), dtype=torch.long),
        batch=torch.zeros(4, dtype=torch.long),
    )

    out = featuriser(data)
    encoder = EMNN(
        input_dim=4,
        edges_input_dim=0,
        width=8,
        width_mult=[1, 1],
        num_vector=2,
        use_auxiliary_frame=False,
        detect_nan=False,
    )

    prepared = encoder._prepare_forward_inputs(out)
    expected_raw_features = torch.cat(
        [out.x.float(), out.canonical_frame.float()], dim=-1
    )
    expected_embedded_features = encoder._embed_node_features(expected_raw_features)

    assert torch.allclose(prepared["x"], expected_embedded_features)
    assert prepared["pos"].shape == (4, 3, 2)


def test_feature_factory_stores_oriented_bbox_frame_separately():
    featuriser = FeatureFactory(
        node_features=["background_binary", "oriented_bbox_frame_cartesian"],
        edge_features=[],
    )

    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.2, 0.7],
            ],
            dtype=torch.float32,
        ),
        vertex_labels=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
        face=torch.tensor(
            [
                [0, 0, 0, 1, 1, 2],
                [1, 2, 3, 2, 4, 3],
                [2, 3, 4, 4, 3, 4],
            ],
            dtype=torch.long,
        ),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )

    out = featuriser(data)

    assert hasattr(out, "canonical_frame")
    assert out.canonical_frame.shape == (5, 3)
    assert out.x.shape == (5, 1)


def test_feature_factory_appends_anchor_distance_features():
    featuriser = FeatureFactory(
        node_features=["background_binary", "com_fps_anchor_distances"],
        edge_features=[],
    )

    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 2.0, 0.5],
                [0.5, 0.2, 1.0],
            ],
            dtype=torch.float32,
        ),
        vertex_labels=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
    )

    out = featuriser(data)
    assert out.x.shape == (5, 5)


def test_new_liver_features_are_finite_and_bounded():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    face = torch.tensor(
        [
            [0, 0],
            [1, 2],
            [2, 3],
        ],
        dtype=torch.long,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
            [1, 0, 2, 1, 3, 2, 0, 3, 2, 0],
        ],
        dtype=torch.long,
    )
    face_adjacency = torch.tensor([[0, 1]], dtype=torch.long)
    face_adjacency_edge_index = torch.tensor([8], dtype=torch.long)

    data = Data(
        pos=pos,
        face=face,
        edge_index=edge_index,
        face_adjacency=face_adjacency,
        face_adjacency_edge_index=face_adjacency_edge_index,
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        face_area=torch.tensor([0.5, 0.5], dtype=torch.float32),
    )

    canonical_cart = compute_canonical_frame(data, coordinates="cartesian")
    canonical_cyl = compute_canonical_frame(data, coordinates="cylindrical")
    oriented_bbox_cart = compute_oriented_bbox_frame(data, coordinates="cartesian")
    oriented_bbox_cyl = compute_oriented_bbox_frame(data, coordinates="cylindrical")
    bbox_cart = compute_bbox_center_coordinates(data, coordinates="cartesian")
    bbox_cyl = compute_bbox_center_coordinates(data, coordinates="cylindrical")
    anchor_distances = compute_com_fps_anchor_distances(data)
    midline_distance = compute_distance_to_bbox_mid_sagittal_plane(data)
    signed_midline = compute_signed_bbox_midline_offset(data)
    vertical_position = compute_bbox_vertical_position(data)
    gaussian = compute_gaussian_curvature(data)
    principal = compute_principal_curvatures(data)
    signed_dihedral = compute_signed_dihedrals(data)
    chiral = compute_chiral_edge_volume(data)
    edge_vertical = compute_edge_vertical_alignment(data)
    edge_midline = compute_edge_midline_proximity(data)

    for tensor in [
        canonical_cart,
        canonical_cyl,
        oriented_bbox_cart,
        oriented_bbox_cyl,
        bbox_cart,
        bbox_cyl,
        anchor_distances,
        midline_distance,
        signed_midline,
        vertical_position,
        gaussian,
        principal,
        signed_dihedral,
        chiral,
        edge_vertical,
        edge_midline,
    ]:
        assert torch.isfinite(tensor).all()

    assert canonical_cart.abs().max().item() <= 1.0 + 1e-6
    assert canonical_cyl[:, 0].abs().max().item() <= 1.0 + 1e-6
    assert canonical_cyl[:, 1].abs().max().item() <= 1.0 + 1e-6
    assert canonical_cyl[:, 2].abs().max().item() <= 1.0 + 1e-6
    assert oriented_bbox_cart.abs().max().item() <= 1.0 + 1e-6
    assert oriented_bbox_cyl[:, 0].abs().max().item() <= 1.0 + 1e-6
    assert oriented_bbox_cyl[:, 1].abs().max().item() <= 1.0 + 1e-6
    assert oriented_bbox_cyl[:, 2].abs().max().item() <= 1.0 + 1e-6
    assert bbox_cart.abs().max().item() <= 1.0 + 1e-6
    assert bbox_cyl[:, 0].abs().max().item() <= 1.0 + 1e-6
    assert bbox_cyl[:, 1].abs().max().item() <= 1.0 + 1e-6
    assert bbox_cyl[:, 2].abs().max().item() <= 1.0 + 1e-6
    assert anchor_distances.min().item() >= 0.0
    assert midline_distance.min().item() >= 0.0
    assert midline_distance.max().item() <= 1.0 + 1e-6
    assert signed_midline.abs().max().item() <= 1.0 + 1e-6
    assert vertical_position.min().item() >= 0.0
    assert vertical_position.max().item() <= 1.0 + 1e-6
    assert gaussian.abs().max().item() <= 1.0 + 1e-6
    assert principal.abs().max().item() <= 1.0 + 1e-6
    assert signed_dihedral.abs().max().item() <= 1.0 + 1e-6
    assert chiral.abs().max().item() <= 1.0 + 1e-6
    assert edge_vertical.min().item() >= 0.0
    assert edge_vertical.max().item() <= 1.0 + 1e-6
    assert edge_midline.min().item() >= 0.0
    assert edge_midline.max().item() <= 1.0 + 1e-6




def test_oriented_bbox_and_anchor_features_are_rotation_stable():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.6],
            [2.0, 0.0, 0.6],
            [2.0, 1.0, 0.6],
            [0.0, 1.0, 0.6],
            [1.7, 0.8, 0.5],
        ],
        dtype=torch.float32,
    )
    face = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 4, 4, 4, 5, 5, 6],
            [1, 2, 3, 2, 4, 3, 5, 6, 7, 6, 8, 7],
            [2, 3, 4, 4, 5, 4, 6, 7, 5, 8, 7, 8],
        ],
        dtype=torch.long,
    )
    data = Data(pos=pos, face=face)

    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotated = Data(pos=pos @ rotation.t(), face=face)

    base_obb = compute_oriented_bbox_frame(data, coordinates="cartesian")
    rotated_obb = compute_oriented_bbox_frame(rotated, coordinates="cartesian")
    base_anchor = compute_com_fps_anchor_distances(data)
    rotated_anchor = compute_com_fps_anchor_distances(rotated)

    assert torch.allclose(base_obb, rotated_obb, atol=1e-4, rtol=1e-4)
    assert torch.allclose(base_anchor, rotated_anchor, atol=1e-4, rtol=1e-4)


def test_stable_canonical_frame_reuses_face_area_if_present():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.2, 1.3],
        ],
        dtype=torch.float32,
    )
    face = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 2, 3, 2],
            [2, 3, 1, 3],
        ],
        dtype=torch.long,
    )
    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    face_area = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

    with_face_area = Data(
        pos=pos,
        face=face,
        face_area=face_area,
    )
    without_face_area = Data(pos=pos, face=face)

    stable_with = compute_stable_canonical_frame(with_face_area, coordinates="cartesian")
    stable_without = compute_stable_canonical_frame(
        without_face_area,
        coordinates="cartesian",
    )

    assert torch.allclose(stable_with, stable_without, atol=1e-5, rtol=1e-5)


def test_stable_canonical_frame_is_rotation_stable():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.4, 0.1, 0.0],
            [2.1, 1.3, 0.2],
            [0.2, 1.5, 0.1],
            [0.4, 0.2, 1.0],
            [2.0, 0.4, 1.1],
            [1.8, 1.1, 1.3],
            [0.3, 1.2, 1.1],
            [1.6, 0.8, 0.7],
        ],
        dtype=torch.float32,
    )
    face = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 3],
            [1, 2, 3, 2, 4, 3, 5, 7, 6, 8, 7, 7],
            [2, 3, 4, 4, 5, 7, 6, 5, 8, 6, 8, 4],
        ],
        dtype=torch.long,
    )
    data = Data(pos=pos, face=face)

    angle_x = torch.tensor(0.37)
    angle_y = torch.tensor(-0.61)
    angle_z = torch.tensor(0.43)
    rot_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(angle_x), -torch.sin(angle_x)],
            [0.0, torch.sin(angle_x), torch.cos(angle_x)],
        ],
        dtype=torch.float32,
    )
    rot_y = torch.tensor(
        [
            [torch.cos(angle_y), 0.0, torch.sin(angle_y)],
            [0.0, 1.0, 0.0],
            [-torch.sin(angle_y), 0.0, torch.cos(angle_y)],
        ],
        dtype=torch.float32,
    )
    rot_z = torch.tensor(
        [
            [torch.cos(angle_z), -torch.sin(angle_z), 0.0],
            [torch.sin(angle_z), torch.cos(angle_z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotation = rot_z @ rot_y @ rot_x
    translated = pos @ rotation.t() + torch.tensor([4.0, -3.0, 1.5], dtype=torch.float32)
    rotated = Data(pos=translated, face=face)

    base_frame = compute_stable_canonical_frame(data, coordinates="cartesian")
    rotated_frame = compute_stable_canonical_frame(rotated, coordinates="cartesian")

    assert torch.allclose(base_frame, rotated_frame, atol=5e-3, rtol=5e-3)




def test_compute_thickness_defaults_to_weighted_average():
    data = Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -2.0],
                [0.1, 0.0, -1.2],
            ],
            dtype=torch.float32,
        ),
        normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )

    thickness, paired_idx = compute_thickness(data, d=1.5, k=4, max_k=4)

    assert torch.isfinite(thickness).all()
    assert torch.all((paired_idx >= 0) & (paired_idx < data.pos.shape[0]))
    assert thickness[0].item() > 1.0
    assert thickness[0].item() < 2.0


def test_compute_thickness_is_batch_safe():
    data_list = [
        Data(
            pos=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, -2.0],
                ],
                dtype=torch.float32,
            ),
            normal=torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        ),
        Data(
            pos=torch.tensor(
                [
                    [10.0, 0.0, 0.0],
                    [10.0, 0.0, -1.5],
                    [10.0, 0.0, -3.0],
                ],
                dtype=torch.float32,
            ),
            normal=torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        ),
    ]
    batch = Data.from_dict({})
    from torch_geometric.data import Batch as PyGBatch

    batch = PyGBatch.from_data_list(data_list)
    thickness, paired_idx = compute_thickness(batch, d=2.0, k=3, max_k=3)

    assert thickness.shape == (6,)
    assert paired_idx.shape == (6,)
    assert torch.all(batch.batch[paired_idx] == batch.batch)
