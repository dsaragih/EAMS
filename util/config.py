from omegaconf import DictConfig, OmegaConf


FEATURE_SIZES: dict[str, int] = {
    # node features
    "pointwise_area": 1,
    "hks": 8,
    "wave_signature": 8,
    "spectral_coordinates": 8,
    "sequence_positional_encoding": 16,
    "background_binary": 1,
    "dental_frame_cartesian": 3,
    "dental_frame_cylindrical": 3,
    "gaussian_curvature": 1,
    "principal_curvatures": 2,
    "canonical_frame_cartesian": 3,
    "canonical_frame_cylindrical": 3,
    "stable_canonical_frame_cartesian": 3,
    "stable_canonical_frame_cylindrical": 3,
    "oriented_bbox_frame_cartesian": 3,
    "oriented_bbox_frame_cylindrical": 3,
    "bbox_center_cartesian": 3,
    "bbox_center_cylindrical": 3,
    "com_fps_anchor_distances": 4,
    "distance_to_bbox_mid_sagittal_plane": 1,
    "signed_bbox_midline_offset": 1,
    "bbox_vertical_position": 1,
    "cartesian_coordinates": 3,
    # edge features
    "degree_weight": 1,
    "anatomy_weight": 1,
    "copy_weight": 1,
    "symmetric_ratios": 2,
    "symmetric_angles": 2,
    "dihedrals": 1,
    "signed_dihedrals": 1,
    "chiral_edge_volume": 1,
    "edge_vertical_alignment": 1,
    "edge_midline_proximity": 1,
    # face features
    "face_di_angle": 1,
    "face_adjacency": 1,
}

EMNN_COORDINATE_NODE_FEATURES = {
    "dental_frame_cartesian",
    "canonical_frame_cartesian",
    "stable_canonical_frame_cartesian",
    "oriented_bbox_frame_cartesian",
}


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def register_custom_omegaconf_resolvers():
    """Register any custom OmegaConf resolvers here."""
    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver(
        "resolve_feature_input_dim",
        lambda features_config, feature_config_name: get_input_dim(
            features_config, feature_config_name
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_emnn_feature_input_dim",
        lambda features_config, feature_config_name, use_auxiliary_frame=True: get_emnn_input_dim(
            features_config, feature_config_name, use_auxiliary_frame
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_emnn_num_vector",
        lambda features_config, base_num_vector, use_auxiliary_frame=True: get_emnn_num_vector(
            features_config, int(base_num_vector), use_auxiliary_frame
        ),
    )


def _get_feature_names(
    features_config: DictConfig,
    feature_config_name: str,
) -> list[str]:
    return list(getattr(features_config, feature_config_name))


def _sum_feature_sizes(feature_names: list[str]) -> int:
    return sum(FEATURE_SIZES[feat] for feat in feature_names)


def get_input_dim(
    features_config: DictConfig,
    feature_config_name: str,
) -> int:
    return _sum_feature_sizes(_get_feature_names(features_config, feature_config_name))


def get_emnn_input_dim(
    features_config: DictConfig,
    feature_config_name: str,
    use_auxiliary_frame: bool = True,
) -> int:
    feature_names = _get_feature_names(features_config, feature_config_name)
    if not _coerce_bool(use_auxiliary_frame):
        return _sum_feature_sizes(feature_names)

    scalar_feature_names = [
        feat for feat in feature_names if feat not in EMNN_COORDINATE_NODE_FEATURES
    ]
    return _sum_feature_sizes(scalar_feature_names)


def get_emnn_num_vector(
    features_config: DictConfig,
    base_num_vector: int = 2,
    use_auxiliary_frame: bool = True,
) -> int:
    if not _coerce_bool(use_auxiliary_frame):
        return base_num_vector

    node_features = _get_feature_names(features_config, "node_features")
    extra_coordinate_channels = sum(
        feat in EMNN_COORDINATE_NODE_FEATURES for feat in node_features
    )
    return base_num_vector + extra_coordinate_channels
