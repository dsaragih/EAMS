from typing import Optional

import torch

# `chamferdist` ships a compiled extension that must match the installed PyTorch.
# On clusters/venvs where it is ABI-incompatible, importing it can fail and would
# otherwise prevent *all* metrics (even Dice) from being importable via Hydra.
try:
    from chamferdist import ChamferDistance as ChamferDistanceLib
except Exception:  # pragma: no cover
    ChamferDistanceLib = None


class DiceCoefficient:
    """Computes Dice coefficient (F1 score) for binary segmentation.

    Measures overlap between predicted and target binary masks.
    Range: [0, 1], where 1 indicates perfect overlap.
    """

    def __init__(self, smooth: float = 1e-6):
        """Initialize Dice coefficient metric.

        Args:
            smooth: Smoothing term to avoid division by zero.
        """
        self.smooth = smooth

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice coefficient between predictions and targets.

        Args:
            predictions: Binary predictions (0 or 1), shape [N] or [B, N]
            targets: Binary ground truth (0 or 1), same shape as predictions

        Returns:
            Dice coefficient as scalar tensor
        """
        predictions = predictions.float()
        targets = targets.float()

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return dice.mean()


class Accuracy:
    """Computes classification accuracy for binary labels.

    Measures percentage of correctly classified samples.
    Range: [0, 1], where 1 indicates perfect classification.
    """

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute accuracy between predictions and targets.

        Args:
            predictions: Binary predictions (0 or 1), shape [N] or [B, N]
            targets: Binary ground truth (0 or 1), same shape as predictions

        Returns:
            Accuracy as scalar tensor
        """
        predictions = predictions.float()
        targets = targets.float()

        correct = (predictions == targets).float().sum()
        total = targets.numel()

        return correct / total


class F1Score:
    """Computes F1 score (harmonic mean of precision and recall).

    Balances precision and recall for binary classification.
    Range: [0, 1], where 1 indicates perfect precision and recall.
    """

    def __init__(self, smooth: float = 1e-6):
        """Initialize F1 score metric.

        Args:
            smooth: Smoothing term to avoid division by zero.
        """
        self.smooth = smooth

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute F1 score between predictions and targets.

        Args:
            predictions: Binary predictions (0 or 1), shape [N] or [B, N]
            targets: Binary ground truth (0 or 1), same shape as predictions

        Returns:
            F1 score as scalar tensor
        """
        predictions = predictions.float()
        targets = targets.float()

        # True positives, false positives, false negatives
        tp = (predictions * targets).sum()
        fp = (predictions * (1 - targets)).sum()
        fn = ((1 - predictions) * targets).sum()

        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)

        f1 = 2 * (precision * recall) / (precision + recall + self.smooth)

        return f1


class IoU:
    """Computes Intersection over Union (Jaccard index).

    Measures overlap between predicted and target regions.
    Range: [0, 1], where 1 indicates perfect overlap.
    """

    def __init__(self, smooth: float = 1e-6):
        """Initialize IoU metric.

        Args:
            smooth: Smoothing term to avoid division by zero.
        """
        self.smooth = smooth

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU between predictions and targets.

        Args:
            predictions: Binary predictions (0 or 1), shape [N] or [B, N]
            targets: Binary ground truth (0 or 1), same shape as predictions

        Returns:
            IoU as scalar tensor
        """
        predictions = predictions.float()
        targets = targets.float()

        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou


class ChamferDistance:
    """Computes bidirectional Chamfer distance between point clouds.

    Measures geometric similarity between predicted and target point sets.
    Lower values indicate better alignment between point clouds.
    """

    def __init__(self, bidirectional: bool = True):
        """Initialize Chamfer distance metric.

        Args:
            bidirectional: If True, compute symmetric Chamfer distance.
        """
        self.bidirectional = bidirectional
        self._use_extension = ChamferDistanceLib is not None
        self.chamfer_dist = ChamferDistanceLib() if self._use_extension else None

    def __call__(
        self, pred_coords: torch.Tensor, gt_coords: torch.Tensor
    ) -> Optional[float]:
        """Compute Chamfer distance between point clouds.

        Args:
            pred_coords: Predicted point cloud, shape [N, 3]
            gt_coords: Ground truth point cloud, shape [M, 3]

        Returns:
            Chamfer distance as float, or None when either class is empty
        """
        # Handle empty cases
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return None

        # Ensure batch dimension
        if pred_coords.ndim == 2:
            pred_coords = pred_coords.unsqueeze(0)
        if gt_coords.ndim == 2:
            gt_coords = gt_coords.unsqueeze(0)

        # Compute distance
        if self._use_extension:
            distance = self.chamfer_dist(
                pred_coords,
                gt_coords,
                bidirectional=self.bidirectional,
                point_reduction="mean",
            )
            return distance.item()

        # Pure PyTorch fallback (slower than chamferdist extension).
        # pred_coords: [B, N, 3], gt_coords: [B, M, 3]
        dists = torch.cdist(pred_coords, gt_coords, p=2)  # [B, N, M]
        # Mean of nearest-neighbour distances
        pred_to_gt = dists.min(dim=2).values.mean(dim=1)  # [B]
        if not self.bidirectional:
            return pred_to_gt.mean().item()
        gt_to_pred = dists.min(dim=1).values.mean(dim=1)  # [B]
        return (pred_to_gt + gt_to_pred).mean().item()


class HausdorffDistance:
    """Computes Hausdorff distance between point clouds.

    Measures maximum distance from any point in one set to the nearest point in the other.
    Sensitive to outliers, useful for worst-case geometric similarity.
    """

    def __call__(
        self, pred_coords: torch.Tensor, gt_coords: torch.Tensor
    ) -> Optional[float]:
        """Compute Hausdorff distance between point clouds.

        Args:
            pred_coords: Predicted point cloud, shape [N, 3]
            gt_coords: Ground truth point cloud, shape [M, 3]

        Returns:
            Hausdorff distance as float, or None when either class is empty
        """
        # Handle empty cases
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return None

        # Ensure 2D tensors
        if pred_coords.ndim == 3:
            pred_coords = pred_coords.squeeze(0)
        if gt_coords.ndim == 3:
            gt_coords = gt_coords.squeeze(0)

        # Compute pairwise distances
        # Shape: [N, M]
        dists = torch.cdist(pred_coords, gt_coords, p=2)

        # Use the symmetric HD95 variant to reduce sensitivity to a few outliers.
        min_pred_to_gt = torch.quantile(dists.min(dim=1).values, 0.95)
        min_gt_to_pred = torch.quantile(dists.min(dim=0).values, 0.95)
        hausdorff = torch.max(min_pred_to_gt, min_gt_to_pred)

        return hausdorff.item()


# Legacy function aliases for backward compatibility
def dice_coefficient(predictions, targets, smooth=1e-6):
    """Legacy function wrapper for DiceCoefficient class."""
    metric = DiceCoefficient(smooth=smooth)
    return metric(predictions, targets)


def _compute_chamfer_distance(pred_coords, gt_coords):
    """Legacy function wrapper for ChamferDistance class."""
    metric = ChamferDistance(bidirectional=True)
    return metric(pred_coords, gt_coords)
