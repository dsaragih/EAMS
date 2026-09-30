"""PyTorch Lightning DataModule for MeshGraphCNN."""

import os

import pytorch_lightning as pl
from typing import List

from data.datasets.segmentation_data import SegmentationData
from data.datasets.classification_data import ClassificationData
from data.datasets.base_dataset import collate_fn
from loguru import logger as log
from torch.utils.data import DataLoader as TorchDataLoader


class MeshDataModule(pl.LightningDataModule):
    """Lightning DataModule for mesh segmentation/classification.

    Wraps the existing SegmentationData and ClassificationData classes
    into PyTorch Lightning's DataModule interface.
    """

    def __init__(
        self,
        dataroot: str,
        folds_dir: str,
        dataset_mode: str = "segmentation",
        data_output_dir: str = "",
        batch_size: int = 16,
        num_workers: int = 4,
        serial_batches: bool = False,
        ninput_edges: int = 22000,
        max_dataset_size: float = float("inf"),
        # Data augmentation
        num_aug: int = 1,
        scale_verts: bool = True,
        slide_verts: float = 0.0,
        flip_edges: float = 0.0,
        transforms: dict | None = None,
        # Model settings (needed for data)
        nclasses: int = 3,
        input_nc: int = 5,
        gpu_ids: list = None,
        seed: int = 2024,
        current_fold: int = 0,
        profile_transforms: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Store parameters
        self.dataroot = dataroot
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.serial_batches = serial_batches
        self.profile_transforms = profile_transforms

        transforms = transforms or {}
        self.pre_transforms = transforms.get("pre_transforms") or []
        self.train_transforms = transforms.get("train_transforms") or []
        self.val_transforms = transforms.get("val_transforms") or []
        self.test_transforms = transforms.get("test_transforms") or []

        # Create a pseudo-opt object for compatibility with existing data classes
        self.opt = type("Options", (), {})()
        self.opt.dataroot = dataroot
        self.opt.dataset_mode = dataset_mode
        self.opt.ninput_edges = ninput_edges
        self.opt.max_dataset_size = max_dataset_size
        self.opt.num_aug = num_aug
        self.opt.scale_verts = scale_verts
        self.opt.slide_verts = slide_verts
        self.opt.flip_edges = flip_edges
        self.opt.nclasses = nclasses
        self.opt.input_nc = input_nc
        self.opt.gpu_ids = gpu_ids if gpu_ids is not None else [0]
        self.opt.serial_batches = serial_batches
        self.opt.seed = seed
        self.opt.export_folder = kwargs.get("export_folder", "")
        self.opt.profile_transforms = profile_transforms

        # Ids are located at split_root/folds/001/train.txt etc.
        split_root = os.path.join(dataroot, folds_dir)
        self.train_ids = os.path.join(split_root, f"{current_fold+1:03d}", "train.txt")
        self.val_ids = os.path.join(split_root, f"{current_fold+1:03d}", "val.txt")
        self.test_ids = os.path.join(split_root, "test.txt")

        # These should be txt files listing the IDs for each phase. verify their existence.
        assert os.path.isfile(
            self.train_ids
        ), f"Train IDs file not found: {self.train_ids}"
        assert os.path.isfile(
            self.val_ids
        ), f"Validation IDs file not found: {self.val_ids}"
        assert os.path.isfile(
            self.test_ids
        ), f"Test IDs file not found: {self.test_ids}"

        if self.pre_transforms:
            log.warning(
                "MeshDataModule ignores pre_transforms for legacy MeshGraphCNN data: {}",
                self.pre_transforms,
            )

    def setup(self, stage: str = None):
        """Setup datasets for each stage."""
        dataset_class = (
            SegmentationData
            if self.dataset_mode == "segmentation"
            else ClassificationData
        )

        if stage == "fit" or stage is None:
            # Training dataset
            self.opt.ids = self.train_ids
            self.opt.phase = "train"
            self.opt.serial_batches = False  # Shuffle for training
            self.train_dataset = dataset_class(
                self.opt,
                transforms=self.train_transforms,
                stage="train",
                profile_transforms=self.profile_transforms,
            )

            # Validation dataset
            self.opt.ids = self.val_ids
            self.opt.phase = "val"
            self.opt.serial_batches = True  # No shuffle for validation
            self.val_dataset = dataset_class(
                self.opt,
                transforms=self.val_transforms,
                stage="val",
                profile_transforms=self.profile_transforms,
            )

        if stage == "test" or stage is None:
            # Test dataset
            self.opt.ids = self.test_ids
            self.opt.phase = "test"
            self.opt.serial_batches = True  # No shuffle for testing
            self.test_dataset = dataset_class(
                self.opt,
                transforms=self.test_transforms,
                stage="test",
                profile_transforms=self.profile_transforms,
            )

    def train_dataloader(self):
        """Create training dataloader."""
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.serial_batches,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def get_nclasses(self):
        """Get number of classes from dataset."""
        if hasattr(self, "train_dataset"):
            return self.train_dataset.nclasses
        return self.opt.nclasses

    def get_input_nc(self):
        """Get number of input channels from dataset."""
        if hasattr(self, "train_dataset"):
            return self.train_dataset.ninput_channels
        return self.opt.input_nc
