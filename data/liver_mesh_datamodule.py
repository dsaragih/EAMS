"""PyTorch Lightning DataModule for MeshGraphCNN."""

import os
import os.path as osp

import pytorch_lightning as pl
from typing import List

from torch_geometric.loader import DataLoader

from data.datasets.liver_mesh_data import LiverMeshDataset
from loguru import logger as log


class LiverMeshDataModule(pl.LightningDataModule):
    """Lightning DataModule for mesh segmentation/classification.

    Wraps the existing SegmentationData and ClassificationData classes
    into PyTorch Lightning's DataModule interface.
    """

    def __init__(
        self,
        dataroot: str,
        folds_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        serial_batches: bool = False,
        # Data transforms
        transforms: dict | None = None,
        features: dict | None = None,
        # Model settings (needed for data)
        nclasses: int = 3,
        current_fold: int = 0,
        profile_transforms: bool = False,
        processed_dir: str | None = None,
        **kwargs,
    ):
        super().__init__()

        # Store parameters
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.serial_batches = serial_batches
        self.n_classes = nclasses
        self.profile_transforms = profile_transforms
        self.processed_dir = (
            osp.abspath(osp.expanduser(processed_dir)) if processed_dir else None
        )
        transforms = transforms or {}
        features = features or {}

        self.pre_transforms = transforms.get("pre_transforms") or []
        self.train_transforms = transforms.get("train_transforms") or []
        self.val_transforms = transforms.get("val_transforms") or []
        self.test_transforms = transforms.get("test_transforms") or []

        # Features will have keys "node_features", "edge_features", etc. combine them
        # NOTE: Featurization now takes place after batching - self.features is often empty
        self.features = []
        for feature_list in features.values():
            self.features.extend(feature_list)

        # Phase settings
        split_root = os.path.join(dataroot, folds_dir)
        self.train_ids = self.open_ids_file(
            os.path.join(split_root, f"{current_fold+1:03d}", "train.txt")
        )
        self.val_ids = self.open_ids_file(
            os.path.join(split_root, f"{current_fold+1:03d}", "val.txt")
        )
        self.test_ids = self.open_ids_file(
            os.path.join(split_root, "test.txt")
        )

    def setup(self, stage: str = None):
        """Setup datasets for each stage."""
        root = os.path.join(self.dataroot, "livermesh")
        edges_root = os.path.join(self.dataroot, "edges")
        labels_root = os.path.join(self.dataroot, "seg")
        common_kwargs = {
            "root": root,
            "edges_root": edges_root,
            "labels_root": labels_root,
            "profile_transforms": self.profile_transforms,
            "file_pattern": "*.obj",
            "pre_transforms": self.pre_transforms,
            "processed_dir": self.processed_dir,
        }

        if stage == "fit" or stage is None:
            log.info("Setting up training and validation datasets...")
            log.info(f"Pre-transforms: {self.pre_transforms}")
            log.info(f"Train transforms: {self.train_transforms}")
            log.info(f"Val transforms: {self.val_transforms}")
            log.info(f"Features: {self.features}")
            # Training dataset
            self.train_dataset = LiverMeshDataset(
                **common_kwargs,
                split="train",
                file_list=self.train_ids,
                transforms=self.train_transforms + self.features,
            )

            # Validation dataset
            self.val_dataset = LiverMeshDataset(
                **common_kwargs,
                split="val",
                file_list=self.val_ids,
                transforms=self.val_transforms + self.features,
            )
            log.info(f"Number of training samples: {len(self.train_dataset)}")
            log.info(f"Number of validation samples: {len(self.val_dataset)}")
        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset = LiverMeshDataset(
                **common_kwargs,
                split="test",
                file_list=self.test_ids,
                transforms=self.test_transforms + self.features,
            )
            log.info(f"Number of test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.serial_batches,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_nclasses(self):
        """Get number of classes from dataset."""
        return self.n_classes

    def get_transform_timing_report(self, stage: str = "train") -> str:
        """Get timing report from the specified dataset.

        Args:
            stage: One of 'train', 'val', or 'test'
        """
        if stage == "train" and hasattr(self, "train_dataset"):
            return self.train_dataset.get_transform_timing_report()
        elif stage == "val" and hasattr(self, "val_dataset"):
            return self.val_dataset.get_transform_timing_report()
        elif stage == "test" and hasattr(self, "test_dataset"):
            return self.test_dataset.get_transform_timing_report()
        else:
            return f"Dataset for stage '{stage}' not found. Make sure setup() has been called."

    def reset_transform_timing(self, stage: str = "train"):
        """Reset timing statistics for the specified dataset.

        Args:
            stage: One of 'train', 'val', or 'test'
        """
        if stage == "train" and hasattr(self, "train_dataset"):
            self.train_dataset.reset_transform_timing()
        elif stage == "val" and hasattr(self, "val_dataset"):
            self.val_dataset.reset_transform_timing()
        elif stage == "test" and hasattr(self, "test_dataset"):
            self.test_dataset.reset_transform_timing()

    def open_ids_file(self, ids_file: str) -> List[str]:
        """Read IDs from a file."""
        if ids_file and os.path.isfile(ids_file):
            with open(ids_file, "r") as f:
                allowed_names = set(line.strip() for line in f if line.strip())
        else:
            allowed_names = None
        return allowed_names


