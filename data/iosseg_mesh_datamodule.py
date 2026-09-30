"""PyTorch Lightning DataModule for IOSSeg 3D PLY mesh data."""

from __future__ import annotations

import os
from typing import List, Optional

import pytorch_lightning as pl
from loguru import logger as log
from torch_geometric.loader import DataLoader

from data.datasets.iosseg_mesh_data import IOSSegMeshDataset
from data.transforms.transforms import IgnoreClassLabels


_SPARSE_MOLAR_CLASS_IDS = (8, 16)  # T18/38 and T28/48


class IOSSegMeshDataModule(pl.LightningDataModule):
    """Lightning DataModule for the IOSSeg 3D dataset (ASCII .ply).

    Conventions mirror `TeethMeshDataModule`:
    - Fixed split files (no k-fold required):
        <dataroot>/<splits_dir>/train_ids.txt
        <dataroot>/<splits_dir>/val_ids.txt
        <dataroot>/<splits_dir>/test_ids.txt

    - Mesh files are loaded from:
        <dataroot>/<meshes_dir>/*.ply

    Split entries can be either mesh ids (e.g. '007_L') or filenames ('007_L.ply').

    Output supervision is on vertices: `batch.vertex_labels`.
    """

    def __init__(
        self,
        dataroot: str,
        splits_dir: str = "splits",
        meshes_dir: str = "train",
        batch_size: int = 16,
        num_workers: int = 4,
        serial_batches: bool = False,
        transforms: Optional[dict] = None,
        features: Optional[dict] = None,
        current_fold: int = 0,
        nclasses: int = 17,
        profile_transforms: bool = False,
        exclude_sparse_molars: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dataroot = dataroot
        self.splits_dir = splits_dir
        self.meshes_dir = meshes_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.serial_batches = serial_batches
        self.nclasses = nclasses
        self.profile_transforms = profile_transforms
        self.label_transforms = (
            [IgnoreClassLabels(_SPARSE_MOLAR_CLASS_IDS)]
            if exclude_sparse_molars
            else []
        )

        transforms = transforms or {}
        features = features or {}

        self.pre_transforms = transforms.get("pre_transforms") or []
        self.train_transforms = transforms.get("train_transforms") or []
        self.val_transforms = transforms.get("val_transforms") or []
        self.test_transforms = transforms.get("test_transforms") or []

        # Combine feature groups (node_features, edge_features, etc.) into a single list.
        self.features: List[str] = []
        for feature_list in features.values():
            self.features.extend(feature_list)

        split_root = os.path.join(self.dataroot, self.splits_dir)
        self.train_ids = self.open_ids_file(
            os.path.join(split_root,  f"{current_fold+1:03d}", "train_ids.txt")
        )
        self.val_ids = self.open_ids_file(os.path.join(split_root, f"{current_fold+1:03d}", "val_ids.txt"))
        self.test_ids = self.open_ids_file(os.path.join(split_root, "test_ids.txt"))

    def setup(self, stage: Optional[str] = None):
        root = os.path.join(self.dataroot, self.meshes_dir)

        common_kwargs = {
            "root": root,
            "profile_transforms": self.profile_transforms,
            "pre_transforms": self.pre_transforms,
        }

        if stage == "fit" or stage is None:
            log.info("Setting up IOSSeg train/val datasets...")
            log.info(f"Root: {root}")
            log.info(f"Pre-transforms: {self.pre_transforms}")
            log.info(f"Train transforms: {self.train_transforms}")
            log.info(f"Val transforms: {self.val_transforms}")
            log.info(f"Features: {self.features}")

            self.train_dataset = IOSSegMeshDataset(
                **common_kwargs,
                split="train",
                file_list=self.train_ids,
                transforms=self.train_transforms + self.features + self.label_transforms,
            )
            self.val_dataset = IOSSegMeshDataset(
                **common_kwargs,
                split="val",
                file_list=self.val_ids,
                transforms=self.val_transforms + self.features + self.label_transforms,
            )
            log.info(f"Number of training samples: {len(self.train_dataset)}")
            log.info(f"Number of validation samples: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            log.info("Setting up IOSSeg test dataset...")
            self.test_dataset = IOSSegMeshDataset(
                **common_kwargs,
                split="test",
                file_list=self.test_ids,
                transforms=self.test_transforms + self.features + self.label_transforms,
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
