import torch
from typing import List, Optional, Union

from torch_geometric.data import Data, Batch
from torch_geometric.nn.encoding import PositionalEncoding

from .node_features import *
from .edge_features import *
from .face_features import *


EMNN_COORDINATE_NODE_FEATURES = {
    "dental_frame_cartesian": "dental_frame",
    "canonical_frame_cartesian": "canonical_frame",
    "stable_canonical_frame_cartesian": "canonical_frame",
    "oriented_bbox_frame_cartesian": "canonical_frame",
}

RIGID_INVARIANT_NODE_FEATURES = {
    "pointwise_area",
    "hks",
    "sequence_positional_encoding",
}

RIGID_INVARIANT_EDGE_FEATURES = {
    "copy_weight",
    "degree_weight",
    "dihedrals",
}

RIGID_INVARIANT_FACE_FEATURES = {
    "face_di_angle",
}


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        node_features: List[str],
        edge_features: List[str],
        face_features: Optional[List[str]] = None,
        others: Optional[List[str]] = None,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.face_features = face_features if face_features is not None else []
        self.others = others if others is not None else []

        if "sequence_positional_encoding" in self.node_features:
            self.positional_encoding = PositionalEncoding(16)

    def _all_features_rigid_transform_invariant(
        self, feature_names: List[str], invariant_features: set[str]
    ) -> bool:
        return all(feature_name in invariant_features for feature_name in feature_names)

    def rigid_transform_invariant_tensors(self) -> dict[str, bool]:
        return {
            "x": self._all_features_rigid_transform_invariant(
                self.node_features, RIGID_INVARIANT_NODE_FEATURES
            ),
            "edge_attr": self._all_features_rigid_transform_invariant(
                self.edge_features, RIGID_INVARIANT_EDGE_FEATURES
            ),
            "face_attr": self._all_features_rigid_transform_invariant(
                self.face_features, RIGID_INVARIANT_FACE_FEATURES
            ),
        }

    def supports_rigid_transform_feature_reuse(self) -> bool:
        if self.others:
            return False

        invariance = self.rigid_transform_invariant_tensors()
        active_tensors = {
            "x": bool(self.node_features),
            "edge_attr": bool(self.edge_features),
            "face_attr": bool(self.face_features),
        }
        return all(
            invariance[name] for name, is_active in active_tensors.items() if is_active
        )

    def compute_node_features(
        self, data: Union[Data, Batch]
    ) -> dict[str, torch.Tensor]:
        feats = []
        out = {}
        for feature_name in self.node_features:
            if feature_name == "pointwise_area":
                area_point, _ = compute_pointwise_area(data)
                feats.append(area_point)
            elif feature_name == "hks":
                # This is slow to compute - check first if computed in pretransforms
                if hasattr(data, "hks"):
                    feats.append(data.hks.float())
                else:
                    feats.append(compute_hks(data))
            elif feature_name == "wave_signature":
                feats.append(compute_wave_signature(data))
            elif feature_name == "spectral_coordinates":
                feats.append(compute_spectral_coordinates(data))
            elif feature_name == "background_binary":
                feats.append(compute_background_binary(data))
            elif feature_name == "gaussian_curvature":
                feats.append(compute_gaussian_curvature(data))
            elif feature_name == "principal_curvatures":
                feats.append(compute_principal_curvatures(data))
            elif feature_name == "bbox_center_cartesian":
                feats.append(compute_bbox_center_coordinates(data, coordinates="cartesian"))
            elif feature_name == "bbox_center_cylindrical":
                feats.append(compute_bbox_center_coordinates(data, coordinates="cylindrical"))
            elif feature_name == "oriented_bbox_frame_cartesian":
                out[EMNN_COORDINATE_NODE_FEATURES[feature_name]] = compute_oriented_bbox_frame(
                    data, coordinates="cartesian"
                )
            elif feature_name == "oriented_bbox_frame_cylindrical":
                feats.append(compute_oriented_bbox_frame(data, coordinates="cylindrical"))
            elif feature_name == "cartesian_coordinates":
                feats.append(data.pos.float())
            elif feature_name == "com_fps_anchor_distances":
                feats.append(compute_com_fps_anchor_distances(data))
            elif feature_name == "distance_to_bbox_mid_sagittal_plane":
                feats.append(compute_distance_to_bbox_mid_sagittal_plane(data))
            elif feature_name == "signed_bbox_midline_offset":
                feats.append(compute_signed_bbox_midline_offset(data))
            elif feature_name == "bbox_vertical_position":
                feats.append(compute_bbox_vertical_position(data))
            elif feature_name == "dental_frame_cartesian":
                out[EMNN_COORDINATE_NODE_FEATURES[feature_name]] = compute_dental_frame(
                    data, coordinates="cartesian"
                )
            elif feature_name == "dental_frame_cylindrical":
                feats.append(compute_dental_frame(data, coordinates="cylindrical"))
            elif feature_name == "canonical_frame_cartesian":
                out[EMNN_COORDINATE_NODE_FEATURES[feature_name]] = compute_canonical_frame(
                    data, coordinates="cartesian"
                )
            elif feature_name == "canonical_frame_cylindrical":
                feats.append(compute_canonical_frame(data, coordinates="cylindrical"))
            elif feature_name == "stable_canonical_frame_cartesian":
                out[EMNN_COORDINATE_NODE_FEATURES[feature_name]] = compute_stable_canonical_frame(
                    data, coordinates="cartesian"
                )
            elif feature_name == "stable_canonical_frame_cylindrical":
                feats.append(
                    compute_stable_canonical_frame(data, coordinates="cylindrical")
                )
            else:
                raise ValueError(f"Node feature '{feature_name}' not recognized.")

        feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
        out["node_features"] = torch.cat(feats, dim=1) if feats else None
        return out

    def compute_edge_features(
        self, data: Union[Data, Batch]
    ) -> dict[str, torch.Tensor]:
        # Edge features
        feats = []
        out = {}
        for feature_name in self.edge_features:
            if feature_name == "degree_weight":
                feats.append(compute_edge_degree_weight(data))
            elif feature_name == "anatomy_weight":
                feats.append(compute_edge_anatomy_weight(data))
            elif feature_name == "copy_weight":
                feats.append(copy_edge_weight(data))
            elif feature_name == "symmetric_ratios":
                feats.append(compute_symmetric_ratios(data))
            elif feature_name == "symmetric_angles":
                feats.append(compute_symmetric_angles(data))
            elif feature_name == "dihedrals":
                feats.append(compute_dihedrals(data))
            elif feature_name == "signed_dihedrals":
                feats.append(compute_signed_dihedrals(data))
            elif feature_name == "chiral_edge_volume":
                feats.append(compute_chiral_edge_volume(data))
            elif feature_name == "edge_vertical_alignment":
                feats.append(compute_edge_vertical_alignment(data))
            elif feature_name == "edge_midline_proximity":
                feats.append(compute_edge_midline_proximity(data))
            else:
                raise ValueError(f"Edge feature '{feature_name}' not recognized.")

        feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
        out["edge_features"] = torch.cat(feats, dim=1) if feats else None
        return out

    def compute_face_features(
        self, data: Union[Data, Batch]
    ) -> dict[str, torch.Tensor]:
        # Face features
        feats = []
        out = {}
        for feature_name in self.face_features:
            if feature_name == "face_di_angle":
                feats.append(compute_di_angle(data))
            else:
                raise ValueError(f"Face feature '{feature_name}' not recognized.")
        feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
        out["face_features"] = torch.cat(feats, dim=1) if feats else None
        return out

    def compute_other_features(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        for feature_name in self.others:
            if feature_name == "vertex2face":

                _, vertex2face = compute_pointwise_area(data)
                data.vertex2face = vertex2face
            elif feature_name == "thickness":
                data.thickness, data.thickness_paired_idx = compute_thickness(data)
            else:
                raise ValueError(f"Other feature '{feature_name}' not recognized.")
        return data

    def forward(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        # Node features
        if self.node_features:
            concat_nf = False
            if hasattr(self, "positional_encoding"):
                data.x = self.positional_encoding(data.seq_pos)
                # This is necessary to concat node features with the positional encoding
                concat_nf = True
            if self.node_features != ["sequence_positional_encoding"]:
                out = self.compute_node_features(data)
                if concat_nf:
                    data.x = torch.cat([data.x, out["node_features"]], dim=-1)
                else:
                    data.x = out["node_features"]

                # Set additional outputs
                for key, value in out.items():
                    if key != "node_features":
                        setattr(data, key, value)
            if data.x is not None:
                data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)

        # Edge features
        if self.edge_features:
            out = self.compute_edge_features(data)
            for key, value in out.items():
                if key == "edge_features":
                    data.edge_attr = torch.nan_to_num(
                        value, nan=0.0, posinf=0.0, neginf=0.0
                    )
                else:
                    setattr(data, key, value)

        # Face features
        if self.face_features:
            out = self.compute_face_features(data)
            for key, value in out.items():
                if key == "face_features":
                    data.face_attr = value
                else:
                    setattr(data, key, value)

        # Other features can be added here
        if self.others:
            data = self.compute_other_features(data)

        return data
