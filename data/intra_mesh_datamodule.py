"""PyTorch Lightning DataModule for IntrA blood vessel (aneurysm) meshes."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from loguru import logger as log
from torch_geometric.loader import DataLoader

from data.datasets.intra_mesh_data import IntrAMeshDataset


class IntrAMeshDataModule(pl.LightningDataModule):
    """Lightning DataModule for IntrA (aneurysm/blood vessel) vertex segmentation.

    Supports k-fold splits similar to LiverMeshDataModule.

    It will first try to read fold files under:
      <dataroot>/<folds_dir>/folds/<FOLD_ID>/train.txt
      <dataroot>/<folds_dir>/folds/<FOLD_ID>/val.txt
      <dataroot>/<folds_dir>/folds/test.txt (optional)

    If those files are missing, it falls back to IntrA's original split files:
      <dataroot>/<splits_subdir>/annSplit_0.txt ... annSplit_{k_folds-1}.txt
    where the selected `current_fold` is used as validation and the remaining
    splits are concatenated as train.
    """

    def __init__(
        self,
        dataroot: str,
        folds_dir: str = "folds/",
        batch_size: int = 16,
        num_workers: int = 4,
        serial_batches: bool = False,
        transforms: Optional[dict] = None,
        features: Optional[dict] = None,
        nclasses: int = 2,
        current_fold: int = 0,
        profile_transforms: bool = False,
        # IntrA layout (preferred: pass explicit roots)
        obj_root: Optional[str] = None,
        label_root: Optional[str] = None,
        gd_root: Optional[str] = None,
        # Backwards-compatible convenience when explicit roots are not passed
        annotated_dir: str = os.path.join("intra", "IntrA", "annotated"),
        obj_subdir: str = "obj",
        label_subdir: str = "ad",
        gd_dir: str = os.path.join("geo", "geo"),
        obj_suffix: str = "_full.obj",
        label_suffix: str = "-_norm.ad",
        gd_suffix: str = "_full.gd",
        use_geodesic_edge_weight: bool = True,
        merge_label_2_to_1: bool = True,
        # IntrA original splits fallback
        splits_subdir: str = os.path.join("splits", "seg"),
        split_prefix: str = "annSplit_",
        k_folds: int = 5,
        **kwargs,
    ):
        super().__init__()

        self.dataroot = dataroot
        self.folds_dir = folds_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.serial_batches = serial_batches
        self.n_classes = nclasses
        self.current_fold = int(current_fold)
        self.profile_transforms = profile_transforms

        transforms = transforms or {}
        features = features or {}

        self.pre_transforms = transforms.get("pre_transforms") or []
        self.train_transforms = transforms.get("train_transforms") or []
        self.val_transforms = transforms.get("val_transforms") or []
        self.test_transforms = transforms.get("test_transforms") or []

        self.features: List[str] = []
        for feature_list in features.values():
            self.features.extend(feature_list)

        self.annotated_dir = annotated_dir
        self.obj_subdir = obj_subdir
        self.label_subdir = label_subdir
        self.gd_dir = gd_dir

        self.obj_root = obj_root
        self.label_root = label_root
        self.gd_root = gd_root

        self.obj_suffix = obj_suffix
        self.label_suffix = label_suffix
        self.gd_suffix = gd_suffix
        self.use_geodesic_edge_weight = bool(use_geodesic_edge_weight)
        self.merge_label_2_to_1 = bool(merge_label_2_to_1)

        self.splits_subdir = splits_subdir
        self.split_prefix = split_prefix
        self.k_folds = int(k_folds)

        self.train_ids, self.val_ids, self.test_ids = self._resolve_splits()

    def _resolve_splits(self) -> Tuple[List[str], List[str], List[str]]:
        # 1) Try fold txt files (Liver-style)
        split_root = os.path.join(self.dataroot, self.folds_dir)
        fold_id = f"{self.current_fold + 1:03d}"

        candidates = [
            (
                os.path.join(split_root, "folds", fold_id, "train.txt"),
                os.path.join(split_root, "folds", fold_id, "val.txt"),
                os.path.join(split_root, "folds", "test.txt"),
            ),
            (
                os.path.join(split_root, fold_id, "train.txt"),
                os.path.join(split_root, fold_id, "val.txt"),
                os.path.join(split_root, "test.txt"),
            ),
        ]

        for train_path, val_path, test_path in candidates:
            if os.path.isfile(train_path) and os.path.isfile(val_path):
                train_ids = self.open_ids_file(train_path)
                val_ids = self.open_ids_file(val_path)
                test_ids = (
                    self.open_ids_file(test_path) if os.path.isfile(test_path) else []
                )
                if not test_ids:
                    log.warning(
                        f"No test split file found at '{test_path}'. Using val split as test."
                    )
                    test_ids = list(val_ids)
                return train_ids, val_ids, test_ids

        # 2) Fallback to IntrA annSplit_*.txt files
        splits_root = os.path.join(self.dataroot, self.splits_subdir)
        split_files: List[str] = []
        for idx in range(self.k_folds):
            p = os.path.join(splits_root, f"{self.split_prefix}{idx}.txt")
            if os.path.isfile(p):
                split_files.append(p)

        if len(split_files) != self.k_folds:
            raise FileNotFoundError(
                "Could not resolve IntrA splits. Tried fold files under "
                f"'{split_root}', and annSplit files under '{splits_root}'."
            )

        val_path = split_files[self.current_fold]
        train_paths = [p for i, p in enumerate(split_files) if i != self.current_fold]

        val_ids = self.open_ids_file(val_path)
        train_ids: List[str] = []
        for p in train_paths:
            train_ids.extend(self.open_ids_file(p))

        # IntrA doesn't define a standard test split; use val by default.
        test_ids = list(val_ids)
        log.warning(
            "IntrA fallback split mode: using validation ids as test ids. "
            "Provide a test.txt under folds if you want a distinct test split."
        )
        return train_ids, val_ids, test_ids

    def setup(self, stage: Optional[str] = None):
        dataset_root = os.path.join(self.dataroot, self.annotated_dir)
        obj_root = (
            self.obj_root
            if self.obj_root is not None
            else os.path.join(dataset_root, self.obj_subdir)
        )
        label_root = (
            self.label_root
            if self.label_root is not None
            else os.path.join(dataset_root, self.label_subdir)
        )
        gd_root = (
            self.gd_root
            if self.gd_root is not None
            else os.path.join(self.dataroot, self.gd_dir)
        )

        fold_tag = f"fold{self.current_fold + 1:03d}"

        common_kwargs = {
            "root": dataset_root,
            "obj_root": obj_root,
            "label_root": label_root,
            "gd_root": gd_root,
            "profile_transforms": self.profile_transforms,
            "pre_transforms": self.pre_transforms,
            "fold_tag": fold_tag,
            "obj_suffix": self.obj_suffix,
            "label_suffix": self.label_suffix,
            "gd_suffix": self.gd_suffix,
            "use_geodesic_edge_weight": self.use_geodesic_edge_weight,
            "merge_label_2_to_1": self.merge_label_2_to_1,
        }

        if stage == "fit" or stage is None:
            log.info("Setting up IntrA train/val datasets...")
            log.info(f"Fold: {self.current_fold}")
            log.info(f"Pre-transforms: {self.pre_transforms}")
            log.info(f"Train transforms: {self.train_transforms}")
            log.info(f"Val transforms: {self.val_transforms}")
            log.info(f"Features: {self.features}")

            self.train_dataset = IntrAMeshDataset(
                **common_kwargs,
                split="train",
                file_list=self.train_ids,
                transforms=self.train_transforms + self.features,
            )
            self.val_dataset = IntrAMeshDataset(
                **common_kwargs,
                split="val",
                file_list=self.val_ids,
                transforms=self.val_transforms + self.features,
            )
            log.info(f"Number of training samples: {len(self.train_dataset)}")
            log.info(f"Number of validation samples: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            log.info("Setting up IntrA test dataset...")
            self.test_dataset = IntrAMeshDataset(
                **common_kwargs,
                split="test",
                file_list=self.test_ids,
                transforms=self.test_transforms + self.features,
            )
            log.info(f"Number of test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.serial_batches,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_transform_timing_report(self, stage: str = "train") -> str:
        if stage == "train" and hasattr(self, "train_dataset"):
            return self.train_dataset.get_transform_timing_report()
        if stage == "val" and hasattr(self, "val_dataset"):
            return self.val_dataset.get_transform_timing_report()
        if stage == "test" and hasattr(self, "test_dataset"):
            return self.test_dataset.get_transform_timing_report()
        return (
            f"Dataset for stage '{stage}' not found. Make sure setup() has been called."
        )

    def reset_transform_timing(self, stage: str = "train"):
        if stage == "train" and hasattr(self, "train_dataset"):
            self.train_dataset.reset_transform_timing()
        elif stage == "val" and hasattr(self, "val_dataset"):
            self.val_dataset.reset_transform_timing()
        elif stage == "test" and hasattr(self, "test_dataset"):
            self.test_dataset.reset_transform_timing()

    @staticmethod
    def open_ids_file(ids_file: str) -> List[str]:
        if ids_file and os.path.isfile(ids_file):
            with open(ids_file, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []
