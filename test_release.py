"""Focused release checks: run with python -m pytest -q test_release.py."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from data.types import MeshData
from data.transforms.transforms import compose_graph_transforms
from util.config import register_custom_omegaconf_resolvers

ROOT = Path(__file__).resolve().parent
RUNS = [
    dict(id='run_01', bundle='INTRA_ARGS emnn_intra sched_plateau aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40_refx'),
    dict(id='run_02', bundle='INTRA_ARGS flat_virtual_emnn_intra sched_plateau invariant_region_centroid aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40_refx'),
    dict(id='run_03', bundle='INTRA_ARGS fast_virtual_emnn_intra sched_plateau_fast aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40_refx'),
    dict(id='run_04', bundle='INTRA_ARGS dgcnn_intra sched_step eams_pointcloud_tcgn_rotation', extra=[], folds=5, evaluation='rz15_rz40_refx'),
    dict(id='run_05', bundle='INTRA_ARGS ptrans_intra eams_pointcloud_tcgn_rotation', extra=[], folds=5, evaluation='rz15_rz40_refx'),
    dict(id='run_06', bundle='IOS_ARGS emnn_teeth sched_plateau normalized_tcgn aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_07', bundle='IOS_ARGS flat_virtual_emnn_teeth sched_plateau invariant_region_centroid normalized_tcgn aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_08', bundle='IOS_ARGS fast_virtual_emnn_teeth sched_plateau_fast normalized_tcgn aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_09', bundle='IOS_ARGS dgcnn_teeth sched_step2 eams_pointcloud_tcgn_rotation', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_10', bundle='IOS_ARGS ptrans_teeth eams_pointcloud_tcgn_rotation', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_11', bundle='IOS_ARGS tcgn_teeth', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_12', bundle='TEETH_ARGS emnn_teeth sched_plateau normalized_tcgn aug_tcgn', extra=[], folds=1, evaluation='rz15_rz40'),
    dict(id='run_13', bundle='TEETH_ARGS flat_virtual_emnn_teeth sched_plateau invariant_region_centroid normalized_tcgn aug_tcgn', extra=[], folds=1, evaluation='rz15_rz40'),
    dict(id='run_14', bundle='TEETH_ARGS fast_virtual_emnn_teeth sched_plateau_fast normalized_tcgn aug_tcgn', extra=[], folds=1, evaluation='rz15_rz40'),
    dict(id='run_15', bundle='TEETH_ARGS tcgn_teeth', extra=[], folds=1, evaluation='rz15_rz40'),
    dict(id='run_16', bundle='LIVER_ARGS emnn_liver sched_plateau2 liver_features_stable liver_local_continuity aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_17', bundle='LIVER_ARGS flat_virtual_emnn_liver sched_plateau2 invariant_region_centroid liver_features_stable liver_virtual_local_continuity aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_18', bundle='LIVER_ARGS fast_virtual_emnn_liver sched_plateau2 liver_features_stable liver_fast_local_continuity aug_tcgn', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_19', bundle='LIVER_ARGS pointnet2_liver eams_pointcloud_tcgn_rotation', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_20', bundle='LIVER_ARGS dgcnn_liver sched_step eams_pointcloud_tcgn_rotation', extra=[], folds=5, evaluation='rz15_rz40'),
    dict(id='run_21', bundle='LIVER_ARGS mesh_graph_cnn sched_plateau2 eams_meshgraphcnn_tcgn_rotation', extra=[], folds=5, evaluation='rz15_rz40'),
]
register_custom_omegaconf_resolvers()


def bundle_args(bundle):
    result = subprocess.run(
        ['bash', '-c', 'set -e; source "$1/emnn_run_helpers.sh"; resolve_group_args "$2" args; printf "%s\\0" "${args[@]}"', 'bash', str(ROOT), bundle],
        check=True, capture_output=True,
    )
    return result.stdout.decode().rstrip('\0').split('\0')


def config(row):
    overrides = ['trainer.max_epochs=100'] + bundle_args(row['bundle']) + row['extra']
    overrides += [f"k_folds={row['folds']}", f"robustness_eval={row['evaluation']}", f'paths.dir_prefix={ROOT}']
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
        return compose(config_name='config', overrides=overrides)


@pytest.mark.parametrize('row', RUNS, ids=[row['id'] for row in RUNS])
def test_main_bundle_composes(row):
    cfg = config(row)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved['data']['dataroot'].startswith(str(ROOT / 'datasets'))
    assert cfg.k_folds == row['folds']
    assert 'tensorboard' in cfg.logger
    for stage in ['val', 'test']:
        compose_graph_transforms(cfg.transforms.get(f'{stage}_transforms') or [], stage=stage)


def test_main_guide_commands_and_override_precedence():
    guide = (ROOT / 'docs/EAMS_EXPERIMENT_RUNS.md').read_text()
    for row in RUNS:
        assert f"`{row['bundle']}`" in guide
    row = dict(RUNS[0], extra=['model.lr=0.0005', 'trainer.max_epochs=150'])
    cfg = config(row)
    assert cfg.model.lr == 0.0005
    assert cfg.trainer.max_epochs == 150


def test_bundle_rejects_unknown_name():
    with pytest.raises(subprocess.CalledProcessError):
        bundle_args('not_a_bundle')


def test_exact_metadata_and_disjoint_splits():
    manifest = json.loads((ROOT / 'docs/SPLIT_MANIFEST.json').read_text())
    def ids(path):
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == len(set(lines)), path
        return set(lines)
    for entry in manifest:
        p = ROOT / entry['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']
        assert len([x for x in p.read_text().splitlines() if x.strip()]) == entry['count']
    for name,folder,suffix in [('All_data','folds',''),('aneurysm_seg','folds',''),('IOSSeg','splits','_ids')]:
        root = ROOT / 'datasets' / name / folder
        tests = ids(root / f'test{suffix}.txt') if name != 'aneurysm_seg' else set()
        vals = []
        for fold in range(1,6):
            f = root / f'{fold:03d}'
            train,val = ids(f / f'train{suffix}.txt'),ids(f / f'val{suffix}.txt')
            assert not train & val
            assert not tests & (train | val)
            if name != "IOSSeg":
                assert not any(val & previous for previous in vals)
            vals.append(val)
        if name == 'aneurysm_seg':
            assert len(set.union(*vals)) == 116
            assert not (root / 'test.txt').exists()
    root = ROOT / 'datasets/Teeth3DS/splits'
    train,val,test = [ids(root / f'{s}_ids.txt') for s in ['train','val','test']]
    assert not (train & val or train & test or val & test)


def test_loader_selects_held_out_intra_validation_and_fixed_teeth():
    from data.intra_mesh_datamodule import IntrAMeshDataModule
    from data.teeth_mesh_datamodule import TeethMeshDataModule
    for fold in range(5):
        dm = IntrAMeshDataModule(str(ROOT / 'datasets/aneurysm_seg'), current_fold=fold)
        assert dm.test_ids == dm.val_ids
        expected = (ROOT / f'datasets/aneurysm_seg/folds/{fold+1:03d}/val.txt').read_text().split()
        assert set(dm.test_ids) == set(expected)
        assert not set(dm.train_ids) & set(dm.test_ids)
    dm0 = TeethMeshDataModule(str(ROOT / 'datasets/Teeth3DS'), current_fold=0)
    dm4 = TeethMeshDataModule(str(ROOT / 'datasets/Teeth3DS'), current_fold=4)
    assert (dm0.train_ids,dm0.val_ids,dm0.test_ids) == (dm4.train_ids,dm4.val_ids,dm4.test_ids)


@pytest.mark.parametrize('encoder', ['emnn_intra','flat_virtual_emnn_intra','fast_virtual_emnn_intra'])
def test_paper_model_forward_contract(encoder):
    row = dict(bundle=f'INTRA_ARGS {encoder} sched_plateau', extra=[], folds=5,evaluation='rz15_rz40_refx')
    cfg = config(row)
    model = instantiate(cfg.model, _recursive_=False).eval()
    n=8
    pairs = torch.tensor([(i,j) for i in range(n) for j in range(n) if i!=j]).T.contiguous()
    mesh = MeshData(pos=torch.randn(n,3), normal=torch.randn(n,3), edge_index=pairs,
                    face=torch.tensor([[0,1,2],[1,2,3],[2,3,4]]).T.contiguous(),
                    x=torch.randn(n,int(cfg.model.encoder_config.input_dim)),
                    edge_attr=torch.randn(pairs.shape[1],int(cfg.model.encoder_config.edges_input_dim)))
    batch = Batch.from_data_list([mesh])
    with torch.no_grad():
        result=model(batch)
    assert isinstance(result,dict)
    logits=result['pred']
    assert logits.shape == (n,2)
    assert torch.isfinite(logits).all()
