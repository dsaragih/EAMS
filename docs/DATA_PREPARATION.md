# Dataset preparation

The release distributes class labels and consumed split metadata only. Obtain the datasets under their original access terms: IntrA from the [authors' project](https://github.com/adrian-0/IntrA), Teeth3DS from the [3DTeethSeg challenge](https://3dteethseg.grand-challenge.org/), 3D-IOSSeg and the annotated liver surfaces from the dataset sources cited in the [paper](https://arxiv.org/abs/2605.08172). CT source data alone does not supply the liver edge annotations. No dataset download, weights, or external annotation tools are bundled.

Place these directories below `paths.dir_prefix` (or set `paths.dataroot` to a different dataset parent). The shipped `.txt` files must stay unchanged:

```text
datasets/
  All_data/
    class_labels.txt
    livermesh/*.obj
    edges/*
    seg/*
    edge_weights/*            # MeshGraphCNN weighting when required
    folds/001/{train,val}.txt # also 002–005
    folds/test.txt
  aneurysm_seg/
    class_labels.txt
    intra/IntrA/annotated/obj/*_full.obj
    intra/IntrA/annotated/ad/*-_norm.ad
    geo/geo/*_full.gd
    folds/001/{train,val}.txt # also 002–005; no shared test.txt
  IOSSeg/
    class_labels.txt
    class_labels_legacy.txt
    all_data/*.ply
    splits/001/{train_ids,val_ids}.txt # also 002–005
    splits/test_ids.txt
  Teeth3DS/
    class_labels.txt
    all_data/<PATIENT>/<PATIENT>_<lower|upper>.obj
    all_data/<PATIENT>/<PATIENT>_<lower|upper>.json
    splits/{train_ids,val_ids,test_ids}.txt
```

IntrA `.ad` vertex labels merge class 2 into class 1. Its `.gd` geodesic matrices supply optional edge weighting. The release has 116 unique IntrA IDs versus the manuscript's 103 meshes: inspect availability and historical preprocessing/accounting before attempting score reproduction. Do not regenerate folds or silently filter IDs. The current loader can omit unavailable raw meshes during processing; check the resulting mesh inventory against every split before training. The release checks verify metadata and loader selection, not this historical count discrepancy.

Liver labels index the provided edge lists; keep mesh vertex ordering, edge files and label files aligned. IOSSeg loads face colors from PLY, maps colors to class IDs, and votes incident face labels onto vertices. Fast-TGCN's adapter preserves its legacy color semantics. Teeth3DS reads JSON vertex labels and maps FDI tooth IDs into the included 17-class label space. Face-level metrics/prediction pooling are implemented in the dental Lightning module. Sparse molars remain included by default.

The included split files have these counts:

| Dataset | Train / validation / test per run |
| --- | --- |
| Liver | 144 / 36 / 20 |
| IntrA | 93 / 23 / 23 for folds 1–4; 92 / 24 / 24 for fold 5 |
| IOSSeg | 110 / 10 / 59 |
| Teeth3DS | 1080 / 120 / 600 |

IntrA's test IDs equal its held-out validation IDs in each fold. Keep the
validation-as-test protocol: do not add a shared `folds/test.txt`.

## Preprocessing and HKS caches

Hydra `transforms.pre_transforms` drives preprocessing automatically on first DataModule setup. It constructs normals, oriented faces, adjacency and vertex-to-edge mappings, then computes/caches spectral signatures or HKS as selected by the preset. `features` is applied after batching in the Lightning module. The default signature implementation uses SciPy's mesh Laplacian and 128 basis functions; LaPy/CHOLMOD are optional backends/accelerators. Cached tensors are version-dependent and must be trusted local artifacts.

Use each run's resolved `transforms` rather than substituting a common pipeline: dental legacy normalization and point-cloud sampling differ from EMNN preprocessing. Do not reuse incompatible preprocessed caches. The deterministic geometry transforms preserve evaluation conventions; randomized baseline point sampling is a known verification limit.

For manual inspection, these dataset utilities expose their actual arguments:

```bash
mesh-venv/bin/python -m data.datasets.liver_mesh_data --help
mesh-venv/bin/python -m data.datasets.teeth_mesh_data --help
mesh-venv/bin/python -m data.datasets.iosseg_mesh_data --help
```

The liver utility accepts explicit `--edges-dir`, `--labels-dir`, `--ids-file` and `--processed-dir`. The Blender utility in `scripts/dataprep/blender_process.py` performs generic decimation when Blender is available; it does not generate anatomical labels or recreate the unpublished historical preparation steps.

[Split counts and SHA-256 hashes](SPLIT_MANIFEST.json) record the exact exported metadata. Liver/IOSSeg testing uses shared held-out files. IntrA testing uses each fold's held-out validation file through the existing fallback. Teeth3DS always selects the one fixed split, irrespective of a `current_fold` keyword.

IOSSeg metadata contains five train/validation resamples (110/10 IDs each) from the same 120-ID training pool, with 59 shared test IDs. Validation sets overlap across runs (40 unique validation IDs in total). They are not five disjoint validation partitions; files are retained exactly.
