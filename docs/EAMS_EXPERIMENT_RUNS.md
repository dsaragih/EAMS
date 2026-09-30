# EAMS experiment guide

## Configuration composition

Run from the repository root with `mesh-venv` activated. Hydra composes
`configs/config.yaml` with dataset, feature, transform, model, encoder, decoder
and loss groups. Runtime paths come from `configs/paths/default.yaml`;
set `paths.dir_prefix` to the directory containing `datasets/` and `experiments/`.
See [dataset preparation](DATA_PREPARATION.md) for data layout and split protocols.

`train_emnn_local.sh` runs locally; `sbatch train_emnn.sh` uses the same bundles
with site-specific SLURM resources. Each quoted argument selects one run.
Bundles expand left to right, followed by overrides after `--`: later Hydra
overrides win. `run_emnn.sh` supplies 100 epochs before bundle expansion.

| Groups | Contribution |
| --- | --- |
| `INTRA_ARGS`, `IOS_ARGS`, `TEETH_ARGS`, `LIVER_ARGS` | Dataset, supervision decoder, features, transforms and fold count. Teeth3DS also sets 50 epochs. |
| `emnn_{intra,teeth,liver}` | EAMS encoder, loss and batch size. |
| `flat_virtual_emnn_{intra,teeth,liver}` | SRA regional context and assignment regularization. |
| `fast_virtual_emnn_{intra,teeth,liver}` | VN virtual-node context and coordinate regularization. |
| `dgcnn_*`, `ptrans_*`, `pointnet2_liver`, `tcgn_teeth`, `mesh_graph_cnn` | Paper baseline encoder/decoder, preprocessing and model-specific training settings. |
| `sched_plateau`, `sched_plateau_fast`, `sched_plateau2` | EAMS plateau schedules; liver uses LR 0.0014, others 0.001. |
| `sched_step`, `sched_step2` | DGCNN step schedules for IntrA/liver and IOSSeg. |
| `invariant_region_centroid` | SRA centroid-distance geometry. |
| `liver_features_stable` | Liver stable PCA Cartesian/cylindrical features. |
| `liver_local_continuity`, `liver_virtual_local_continuity`, `liver_fast_local_continuity` | Architecture-specific liver continuity loss. |
| `normalized_tcgn`, `aug_tcgn` | Legacy normalization, then bounded 27° rotation augmentation and normalization. |
| `eams_pointcloud_tcgn_rotation`, `eams_meshgraphcnn_tcgn_rotation` | Matched bounded rotation augmentation with baseline preprocessing. |

For example, change learning rate and epoch count directly:

```bash
./train_emnn_local.sh "INTRA_ARGS emnn_intra sched_plateau aug_tcgn" -- model.lr=0.0005 trainer.max_epochs=150 paths.dir_prefix=/path/to/eams-data
```

Append `--cfg job --resolve` after `--` to inspect the resolved configuration.
All encoder/decoder and other Hydra configs remain available for direct overrides,
including newer canonicalization and frame-averaging architectures. When selecting
those architectures, compose their compatible encoder, decoder and loss configs.

## Main comparisons

Use the bundle in each row with `./train_emnn_local.sh "BUNDLE" -- paths.dir_prefix=/path/to/eams-data`.
IntrA uses five validation-as-test folds; IOSSeg and liver use five training
folds with shared held-out tests. Teeth3DS uses one fixed split.
The commands specify the release's main comparison configurations; exact
historical scores are not guaranteed. EAMS-family runs use matched training
rotation augmentation. Dental EAMS runs retain the explicit normalization
modifier, with the final augmentation bundle selecting the combined pipeline.

| Dataset | Model | Bundle |
| --- | --- | --- |
| IntrA | EAMS | `INTRA_ARGS emnn_intra sched_plateau aug_tcgn` |
| IntrA | SRA+EAMS | `INTRA_ARGS flat_virtual_emnn_intra sched_plateau invariant_region_centroid aug_tcgn` |
| IntrA | VN+EAMS | `INTRA_ARGS fast_virtual_emnn_intra sched_plateau_fast aug_tcgn` |
| IntrA | DGCNN | `INTRA_ARGS dgcnn_intra sched_step eams_pointcloud_tcgn_rotation` |
| IntrA | PTv3 | `INTRA_ARGS ptrans_intra eams_pointcloud_tcgn_rotation` |
| IOSSeg | EAMS | `IOS_ARGS emnn_teeth sched_plateau normalized_tcgn aug_tcgn` |
| IOSSeg | SRA+EAMS | `IOS_ARGS flat_virtual_emnn_teeth sched_plateau invariant_region_centroid normalized_tcgn aug_tcgn` |
| IOSSeg | VN+EAMS | `IOS_ARGS fast_virtual_emnn_teeth sched_plateau_fast normalized_tcgn aug_tcgn` |
| IOSSeg | DGCNN | `IOS_ARGS dgcnn_teeth sched_step2 eams_pointcloud_tcgn_rotation` |
| IOSSeg | PTv3 | `IOS_ARGS ptrans_teeth eams_pointcloud_tcgn_rotation` |
| IOSSeg | Fast-TGCN | `IOS_ARGS tcgn_teeth` |
| Teeth3DS | EAMS | `TEETH_ARGS emnn_teeth sched_plateau normalized_tcgn aug_tcgn` |
| Teeth3DS | SRA+EAMS | `TEETH_ARGS flat_virtual_emnn_teeth sched_plateau invariant_region_centroid normalized_tcgn aug_tcgn` |
| Teeth3DS | VN+EAMS | `TEETH_ARGS fast_virtual_emnn_teeth sched_plateau_fast normalized_tcgn aug_tcgn` |
| Teeth3DS | Fast-TGCN | `TEETH_ARGS tcgn_teeth` |
| Liver | EAMS | `LIVER_ARGS emnn_liver sched_plateau2 liver_features_stable liver_local_continuity aug_tcgn` |
| Liver | SRA+EAMS | `LIVER_ARGS flat_virtual_emnn_liver sched_plateau2 invariant_region_centroid liver_features_stable liver_virtual_local_continuity aug_tcgn` |
| Liver | VN+EAMS | `LIVER_ARGS fast_virtual_emnn_liver sched_plateau2 liver_features_stable liver_fast_local_continuity aug_tcgn` |
| Liver | PointNet++ | `LIVER_ARGS pointnet2_liver eams_pointcloud_tcgn_rotation` |
| Liver | DGCNN | `LIVER_ARGS dgcnn_liver sched_step eams_pointcloud_tcgn_rotation` |
| Liver | MeshGraphCNN | `LIVER_ARGS mesh_graph_cnn sched_plateau2 eams_meshgraphcnn_tcgn_rotation` |

EAMS-family runs use 100 epochs except Teeth3DS (50); baseline bundles set
200–400 epochs and their own batch sizes and optimizers. EAMS uses AdamW;
PointNet++, PTv3 and Fast-TGCN select Adam. Encoder defaults use GraphNorm;
liver main bundles explicitly select batch normalization. Inspect the resolved
config for loss coefficients, token counts and decoder settings.

## Evaluation

Training saves the best validation-IoU checkpoint and runs configured testing.
Reevaluate saved experiments with:

```bash
mesh-venv/bin/python test_experiment.py /path/to/experiment --robustness-eval-config rz15_rz40
mesh-venv/bin/python test_experiment.py /path/to/intra-experiment --robustness-eval-config rz15_rz40_refx
mesh-venv/bin/python test_experiment.py /path/to/experiment --fold 0 --robustness-eval-config random_rottrans_eval
mesh-venv/bin/python evaluate_pca_stability.py --experiment /path/to/experiment --num-rotations 32 --seed 2024
mesh-venv/bin/python evaluate_liver_eams_failures.py /path/to/liver-experiment
```

Fold indices are zero-based. Saved configs contain resolved paths; update them
when moving experiments. `rz15_rz40` tests fixed rotations; the IntrA preset adds
reflection. Dental reflection changes left/right labels and is outside the target
group. `random_rottrans_eval` uses one seed-42 sample-specific random-axis
rotation bounded by 90°, without translation. PCA diagnostics measure empirical
frame and prediction stability. Point-cloud baseline sampling remains randomized,
so repeated evaluation can vary even with fixed geometric transforms.
