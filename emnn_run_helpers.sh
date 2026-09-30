#!/bin/bash
# Main EAMS comparison bundles. Compose left to right; later Hydra overrides win.
# Additional architectures and settings remain available through configs/ overrides.

DEFAULT_RUN_GROUPS=("INTRA_ARGS emnn_intra sched_plateau aug_tcgn")

# Dataset selection
IOS_ARGS=(
    "model=emnn_teeth"
    "model/decoder_config=vertex_emnn"
    "data=iosseg_graph"
    "features=vertex_hks_dihedrals_frame_cyl"
    "transforms=idx_no_aug_hks"
    "robustness_eval=rz15_rz40"
    "k_folds=5"
)

TEETH_ARGS=(
    "model=emnn_teeth"
    "model/decoder_config=vertex_emnn"
    "data=teeth_graph"
    "features=vertex_hks_dihedrals_frame_cyl"
    "transforms=idx_no_aug_hks"
    "robustness_eval=rz15_rz40"
    "k_folds=1"
    "trainer.max_epochs=50"
)

INTRA_ARGS=(
    "model=emnn_intra"
    "model/decoder_config=vertex_emnn"
    "data=intra_graph"
    "features=vertex_hks_dihedrals"
    "transforms=idx_no_aug_hks"
    "k_folds=5"
)

LIVER_ARGS=(
    "model=emnn_liver"
    "model/decoder_config=edge_emnn"
    "data=liver_graph"
    "features=edge_hks_dihedrals"
    "transforms=idx_no_aug_hks"
    "robustness_eval=rz15_rz40"
    "k_folds=5"
)


# EAMS architectures
emnn_intra=(
    "model/encoder_config=vertex_emnn"
    "model/loss_setup=intra_vbdou_contrastive"
    "model.loss_setup.loss_coefficients.embedding=10.0"
    "data.batch_size=1"
)

flat_virtual_emnn_intra=(
    "model/encoder_config=vertex_virtual_emnn"
    "model/loss_setup=intra_vbdou_contrast_virtual"
    "model.loss_setup.loss_coefficients.embedding=10.0"
    "model.loss_setup.loss_coefficients.assignment_matrix=1.0"
    "data.batch_size=1"
)

fast_virtual_emnn_intra=(
    "model/encoder_config=vertex_fast_virtual_emnn"
    "model/loss_setup=intra_vbdou_contrast_fast_virtual"
    "model.loss_setup.loss_coefficients.embedding=10.0"
    "model.loss_setup.loss_coefficients.virtual_node_coords=1.0"
    "data.batch_size=1"
)

emnn_teeth=(
    "model/encoder_config=vertex_emnn"
    "model/loss_setup=teeth_vbdou_contrastive"
    "model.loss_setup.loss_coefficients.embedding=10.0"
    "data.batch_size=1"
)

flat_virtual_emnn_teeth=(
    "model/encoder_config=vertex_virtual_emnn"
    "model/loss_setup=teeth_vbdou_contrast_virtual"
    "model.loss_setup.loss_coefficients.embedding=10.0"
    "model.loss_setup.loss_coefficients.assignment_matrix=1.0"
    "data.batch_size=1"
)

fast_virtual_emnn_teeth=(
    "model/encoder_config=vertex_fast_virtual_emnn"
    "model/loss_setup=teeth_vbdou_contrast_fast_virtual"
    "model.loss_setup.loss_coefficients.embedding=10.0"
    "model.loss_setup.loss_coefficients.virtual_node_coords=1.0"
    "data.batch_size=1"
)

emnn_liver=(
    "model/encoder_config=edge_emnn"
    "model.encoder_config.virtual_global_node=true"
    "model.encoder_config.use_radial_edge_embedding=true"
    "model.encoder_config.conv_attention=true"
    "model.encoder_config.norm_type=bn"
    "model.decoder_config.dropout=0.1"
    "model/loss_setup=liver_vbdou_contrastive"
    "model.loss_setup.loss_coefficients.embedding=1.0"
    "data.batch_size=8"
    "model.encoder_config.thickness_scale=3.0"
    "model.encoder_config.thickness_init=1.0"
)

flat_virtual_emnn_liver=(
    "model/encoder_config=edge_virtual_emnn"
    "model/loss_setup=liver_vbdou_contrast_virtual"
    "model.loss_setup.loss_coefficients.embedding=1.0"
    "model.encoder_config.norm_type=bn"
    "model.decoder_config.dropout=0.1"
    "model.loss_setup.loss_coefficients.assignment_matrix=1.0"
    "data.batch_size=8"
    "model.encoder_config.thickness_scale=3.0"
    "model.encoder_config.thickness_init=1.0"
)

fast_virtual_emnn_liver=(
    "model/encoder_config=edge_fast_virtual_emnn"
    "model/loss_setup=liver_vbdou_contrast_fast_virtual"
    "model.loss_setup.loss_coefficients.embedding=1.0"
    "model.encoder_config.norm_type=bn"
    "model.decoder_config.dropout=0.1"
    "model.loss_setup.loss_coefficients.virtual_node_coords=1.0"
    "data.batch_size=8"
    "model.encoder_config.thickness_scale=3.0"
    "model.encoder_config.thickness_init=1.0"
    "model.encoder_config.num_fast_virtual_nodes=8"
)


# Paper baselines
dgcnn_intra=(
    "model/encoder_config=dgcnn_vertex"
    "model/decoder_config=mlp_vertex"
    "model/loss_setup=ce_unif"
    "transforms=pointcloud_baseline"
    "data.batch_size=8"
    "trainer.max_epochs=400"
)

ptrans_intra=(
    "model/encoder_config=ptv3_vertex"
    "model/decoder_config=mlp_vertex"
    "model.optimizer_name=adam"
    "model.lr=0.001"
    "model.weight_decay=1.0e-4"
    "model.lr_policy=step"
    "model.lr_decay_iters=20"
    "model.lr_step_gamma=0.5"
    "transforms=pointcloud_baseline"
    "data.batch_size=4"
    "trainer.max_epochs=200"
)

dgcnn_teeth=(
    "model/encoder_config=dgcnn_vertex"
    "model/decoder_config=mlp_face"
    "model/loss_setup=teeth_ce"
    "transforms=pointcloud_baseline"
    "transforms.n_points=8192"
    "data.batch_size=8"
    "trainer.max_epochs=250"
)

ptrans_teeth=(
    "model/encoder_config=ptv3_vertex"
    "model/decoder_config=mlp_face"
    "model/loss_setup=teeth_ce"
    "model.optimizer_name=adam"
    "model.lr=0.001"
    "model.weight_decay=1.0e-4"
    "model.lr_policy=step"
    "model.lr_decay_iters=20"
    "model.lr_step_gamma=0.5"
    "transforms=pointcloud_baseline"
    "transforms.n_points=8192"
    "data.batch_size=4"
    "trainer.max_epochs=200"
)

tcgn_teeth=(
    "model/encoder_config=vertex_tcgn"
    "model/decoder_config=mlp_face"
    "model/loss_setup=teeth_ce"
    "model.optimizer_name=adam"
    "model.lr=0.001"
    "model.weight_decay=1.0e-5"
    "model.lr_policy=step"
    "model.lr_decay_iters=20"
    "model.lr_step_gamma=0.5"
    "transforms=idx_tcgn_legacy"
    "data.batch_size=1"
    "trainer.max_epochs=200" # 300
)

pointnet2_liver=(
    "model/encoder_config=pointnet2_msg_vertex"
    "model/decoder_config=edge_emnn"
    "model.optimizer_name=adam"
    "model.lr=0.001"
    "model.weight_decay=1.0e-4"
    "model.lr_policy=step"
    "model.lr_decay_iters=20"
    "model.lr_step_gamma=0.5"
    "transforms=pointcloud_baseline"
    "data.batch_size=8"
    "trainer.max_epochs=200"
)

dgcnn_liver=(
    "model/encoder_config=dgcnn_vertex"
    "model/decoder_config=edge_emnn"
    "transforms=pointcloud_baseline"
    "data.batch_size=8"
    "trainer.max_epochs=300"
)

mesh_graph_cnn=(
    "model=mesh_encoder_decoder_refactored"
    "data=liver_mesh"
)


# Schedules
sched_plateau=(
    "model.lr=0.001"
    "model.lr_policy=plateau"
)

sched_plateau2=(
    "model.lr=0.0014"
    "model.lr_policy=plateau"
    "model.plateau_factor=0.6"
    "model.plateau_threshold=0.01"
    "model.plateau_patience=2"
)

sched_plateau_fast=(
    "model.lr=0.001"
    "model.lr_policy=plateau"
    "model.plateau_factor=0.8"
    "model.plateau_patience=1"
)

sched_step=(
    "model.lr=0.001"
    "model.lr_policy=step"
    "model.lr_decay_iters=30"
    "model.lr_step_gamma=0.7"
)

sched_step2=(
    "model.lr=0.001"
    "model.lr_policy=step"
    "model.lr_decay_iters=20"
    "model.lr_step_gamma=0.7"
)


# Preprocessing, loss and augmentation modifiers
invariant_region_centroid=(
    "model.encoder_config.regional_use_coordinate_features=false"
    "model.encoder_config.regional_geometry_feature_mode=centroid_distance"
    "model.encoder_config.regional_use_region_geometry_features=true"
)

normalized_tcgn=(
    "transforms=idx_tcgn_legacy_norm"
)

liver_features_stable=(
    "features=edge_hks_stable_canonical_frame_cartcyl"
    "model.encoder_config.use_auxiliary_frame=false"
)

liver_local_continuity=(
    "model/loss_setup=liver_vbdou_contrastive_continuity"
    "model.loss_setup.loss_coefficients.continuity=1.0"
)

liver_virtual_local_continuity=(
    "model/loss_setup=liver_vbdou_contrast_virtual_continuity"
    "model.loss_setup.loss_coefficients.continuity=1.0"
)

liver_fast_local_continuity=(
    "model/loss_setup=liver_vbdou_contrast_fast_virtual_continuity"
    "model.loss_setup.loss_coefficients.continuity=1.0"
)

aug_tcgn=(
    "transforms=idx_tcgn_legacy"
)

eams_pointcloud_tcgn_rotation=(
    "transforms=pointcloud_tcgn_rotation"
)

eams_meshgraphcnn_tcgn_rotation=(
    "transforms=legacy_tcgn_rotation"
)

validate_config_bundle() {
    local config_name="$1"
    if ! declare -p "$config_name" >/dev/null 2>&1; then
        echo "Unknown config bundle: $config_name" >&2
        return 1
    fi
}

resolve_group_args() {
    local group_spec="$1"
    local -n output_args_ref="$2"
    local config_name
    local -a group_names=()
    local -a bundle_args=()

    output_args_ref=()
    read -r -a group_names <<< "$group_spec"

    if [[ "${#group_names[@]}" -eq 0 ]]; then
        echo "Empty config group is not allowed" >&2
        return 1
    fi

    for config_name in "${group_names[@]}"; do
        validate_config_bundle "$config_name" || return 1
        declare -n config_args_ref="$config_name"
        bundle_args+=("${config_args_ref[@]}")
    done

    output_args_ref=("${bundle_args[@]}")
}

format_group_label() {
    local group_spec="$1"
    local -a group_names=()
    local joined_label=""

    read -r -a group_names <<< "$group_spec"
    joined_label="${group_names[*]}"
    echo "${joined_label// /+}"
}
