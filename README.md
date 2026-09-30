# EAMS: anatomical mesh segmentation

Code for **Augmented Equivariant Mesh Networks for Anatomical Segmentation**, by Daniel Saragih ([arXiv](https://arxiv.org/abs/2605.08172)). EAMS uses EMNN encoders with edge, vertex, or face supervision. SRA and VN variants add regional or virtual-node context.

This release contains code and original split metadata. Download datasets separately;
weights and generated results are excluded. The [experiment guide](docs/EAMS_EXPERIMENT_RUNS.md)
explains configuration composition, main comparisons and evaluation.

## Installation

The release checks use Python 3.11.5, PyTorch 2.9.1, CUDA 12.6, PyTorch Lightning 2.6.0 and an NVIDIA H100. Install a PyTorch build compatible with your machine first, then matching [PyG extensions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html):

```bash
python3.11 -m venv mesh-venv
source mesh-venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.9.0+cu126.html
python -m pip install -r requirements.txt
python -m pytest -q test_release.py test_runtime_transforms.py test_emnn_dental_frame.py test_stable_pca_diagnostics.py
```

Some platforms require building torch-scatter/torch-cluster with the matching CUDA toolkit. LaPy enables the optional FEM backend; scikit-sparse/CHOLMOD accelerates eigensolves and needs SuiteSparse development libraries. DGCNN, Fast-TGCN and PointNet++ use the included adapters; PTv3 additionally needs [optional requirements](requirements-baselines.txt); its CUDA wheel selection follows [spconv installation](https://github.com/traveller59/spconv#install). W&B is optional; use `logger=wandb` after installing it. TensorBoard is the credential-free default.

## Data and preprocessing

Follow [data preparation](docs/DATA_PREPARATION.md) to obtain data and place meshes/labels beside the included metadata. `paths.dir_prefix` is the directory containing `datasets/` and `experiments/`; it defaults to the current directory and can also be set with `EAMS_ROOT`. Data roots, edge weights, checkpoints and results are configured in `configs/paths/default.yaml`. Preprocessing runs on first dataset setup, caches HKS and geometry, and may take substantially longer than training.

## Training

Run from the repository root, with the environment activated:

```bash
./train_emnn_local.sh "INTRA_ARGS emnn_intra sched_plateau aug_tcgn" -- paths.dir_prefix=/path/to/eams-data robustness_eval=rz15_rz40_refx
```

Each quoted argument selects one run and combines the named bundles. Later Hydra
overrides win. See the [experiment guide](docs/EAMS_EXPERIMENT_RUNS.md) for the
main EAMS/SRA/VN and baseline commands, configuration inspection and evaluation.
`train_emnn.sh` provides generic SLURM directives; adapt resources and load your
site's modules before submission.

## Citation and licensing

```bibtex
@article{saragih2026eams,
  title={Augmented Equivariant Mesh Networks for Anatomical Segmentation},
  author={Saragih, Daniel},
  year={2026},
  eprint={2605.08172},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

MIT; see [LICENSE](LICENSE), [CITATION.cff](CITATION.cff) and [third-party notices](THIRD_PARTY_NOTICES.md). This repository has independent release history.
