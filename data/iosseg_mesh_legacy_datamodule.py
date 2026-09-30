"""PyTorch Lightning DataModule for raw-Ply IOSSeg legacy TCGN parity."""

from __future__ import annotations

import os
from typing import List, Optional

import pytorch_lightning as pl
from loguru import logger as log
from torch_geometric.loader import DataLoader

from data.datasets.iosseg_mesh_legacy_data import IOSSegLegacyMeshDataset


class IOSSegLegacyMeshDataModule(pl.LightningDataModule):
    """IOSSeg DataModule that bypasses preprocessing cache and pre-transforms."""

    def __init__(
        self,
        dataroot: str,
        splits_dir: str = "splits",
        meshes_dir: str = "train",
        batch_size: int = 16,
        num_workers: int = 4,
        serial_batches: bool = False,
        transforms: Optional[dict] = None,
        nclasses: int = 17,
        profile_transforms: bool = False,
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

        transforms = transforms or {}
        self.pre_transforms = transforms.get("pre_transforms") or []
        self.train_transforms = transforms.get("train_transforms") or []
        self.val_transforms = transforms.get("val_transforms") or []
        self.test_transforms = transforms.get("test_transforms") or []

        split_root = os.path.join(self.dataroot, self.splits_dir)
        self.train_ids = self.open_ids_file(os.path.join(split_root, "train_ids.txt"))
        self.val_ids = self.open_ids_file(os.path.join(split_root, "val_ids.txt"))
        self.test_ids = self.open_ids_file(os.path.join(split_root, "test_ids.txt"))

        if self.pre_transforms:
            log.warning(
                "IOSSegLegacyMeshDataModule ignores pre_transforms to preserve raw PLY parity: {}",
                self.pre_transforms,
            )

    def setup(self, stage: Optional[str] = None):
        root = os.path.join(self.dataroot, self.meshes_dir)

        if stage == "fit" or stage is None:
            log.info("Setting up raw IOSSeg legacy train/val datasets...")
            log.info(f"Root: {root}")
            log.info(f"Train transforms: {self.train_transforms}")
            log.info(f"Val transforms: {self.val_transforms}")

            self.train_dataset = IOSSegLegacyMeshDataset(
                root=root,
                split="train",
                file_list=self.train_ids,
                transforms=self.train_transforms,
                profile_transforms=self.profile_transforms,
            )
            self.val_dataset = IOSSegLegacyMeshDataset(
                root=root,
                split="val",
                file_list=self.val_ids,
                transforms=self.val_transforms,
                profile_transforms=self.profile_transforms,
            )

        if stage == "test" or stage is None:
            log.info("Setting up raw IOSSeg legacy test dataset...")
            log.info(f"Test transforms: {self.test_transforms}")

            self.test_dataset = IOSSegLegacyMeshDataset(
                root=root,
                split="test",
                file_list=self.test_ids,
                transforms=self.test_transforms,
                profile_transforms=self.profile_transforms,
            )

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
