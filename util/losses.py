from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# from loguru import logger
# import time


# def get_loss(loss_name: str, label_smoothing: float = 0.0):
#     """Return loss function based on the loss name."""
#     # Dictionary mapping loss names to loss functions
#     loss_name_dict = {
#         "ce": lambda pred, target: F.cross_entropy(
#             pred, target, label_smoothing=label_smoothing
#         ),
#         "mse": lambda pred, target: F.mse_loss(pred, target),
#         "bce": lambda pred, target: F.binary_cross_entropy_with_logits(pred, target),
#         "dice_loss": lambda pred, target: 1 - dice_coefficient(pred, target),
#         # Add more loss functions as needed
#     }

#     def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """Compute combined loss based on parsed expression."""
#         # Replace loss names with function calls
#         expression = loss_name
#         result = None

#         # Parse and evaluate the expression
#         tokens = (
#             expression.replace("+", " + ")
#             .replace("-", " - ")
#             .replace("*", " * ")
#             .replace("/", " / ")
#             .split()
#         )

#         stack = []
#         i = 0
#         while i < len(tokens):
#             token = tokens[i].strip()
#             if not token:
#                 i += 1
#                 continue

#             if token in loss_name_dict:
#                 # Compute the loss
#                 stack.append(loss_name_dict[token](pred, target))
#             elif token in ["+", "-", "*", "/"]:
#                 stack.append(token)
#             else:
#                 # Try to parse as a number (coefficient)
#                 try:
#                     stack.append(float(token))
#                 except ValueError:
#                     pass
#             i += 1

#         # Evaluate the stack
#         if not stack:
#             raise ValueError(f"Invalid loss name: {loss_name}")

#         result = stack[0]
#         i = 1
#         while i < len(stack):
#             if i + 1 < len(stack):
#                 op = stack[i]
#                 operand = stack[i + 1]
#                 if op == "+":
#                     result = result + operand
#                 elif op == "-":
#                     result = result - operand
#                 elif op == "*":
#                     result = result * operand
#                 elif op == "/":
#                     result = result / operand
#                 i += 2
#             else:
#                 i += 1

#         return result

#     return combined_loss


###### Loss Classes ######


class BaseLoss:
    # Base class for loss functions
    # Has an is_weighted_loss attribute
    # Has a required_attributes list for data attributes to pass into kwargs
    is_weighted_loss: bool = False
    requires_supervision: bool = True

    @property
    def supervision_key(self) -> str | None:
        return None

    @property
    def model_output_key(self) -> str | None:
        return None

    @property
    def required_attributes(self) -> List[str]:
        return []

    def __call__(
        self,
        model_output: torch.Tensor,
        supervision: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("BaseLoss is an abstract class.")


class DiceLoss(BaseLoss):
    def __init__(self, smooth: float = 1e-6):
        self.smooth = smooth
        self.is_weighted_loss = False

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return dice_loss_standard(predictions, targets, self.smooth)


class CrossEntropyLoss(BaseLoss):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_weights: List[float] = None,
        batched_input: bool = True,
    ):
        self.label_smoothing = label_smoothing
        self.label_weights = (
            torch.tensor(label_weights) if label_weights is not None else None
        )
        self.is_weighted_loss = True
        self.batched_input = batched_input

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        self.label_weights = (
            self.label_weights.to(predictions.device)
            if self.label_weights is not None
            else None
        )
        return F.cross_entropy(
            predictions,
            targets,
            weight=self.label_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,
        )


class MeshGraphCNNLoss(BaseLoss):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_weights: List[float] = None,
        batched_input: bool = True,
    ):
        self.label_smoothing = label_smoothing
        self.label_weights = (
            torch.tensor(label_weights) if label_weights is not None else None
        )
        self.is_weighted_loss = True
        self.batched_input = batched_input

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        w_loss: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if w_loss is None:
            w_loss = torch.ones_like(predictions, dtype=torch.float32)
        self.label_weights = (
            self.label_weights.to(predictions.device)
            if self.label_weights is not None
            else None
        )
        loss = F.cross_entropy(
            predictions,
            targets,
            weight=self.label_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,
        )

        if not self.batched_input:
            dice = dice_loss_new(predictions, targets, w_loss)
        else:
            dice = dice_loss_new_original(predictions, targets, w_loss)
        loss = loss + dice
        return loss


class GraphCutLoss(BaseLoss):
    """
    Graph cut loss that penalizes prediction disagreement between adjacent edges
    with different ground-truth labels.

    Loss formula: CE + sum_{(e1,e2) in P} w_uv * (1 - 0.5 * ||p_e1 - p_e2||_1) * I(y_e1 != y_e2)

    Args:
        batched_input: Whether input is in batched PyG format (flattened across batch)
        angle_weights: Whether to use angle-based weights (will ignore w_loss if True)
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_weights: List[float] = None,
        ce_scalar: float = 1.0,
        batched_input: bool = True,
        angle_weights: bool = False,
    ):
        self.label_smoothing = label_smoothing
        self.label_weights = (
            torch.tensor(label_weights) if label_weights is not None else None
        )
        self.ce_scalar = ce_scalar
        self.batched_input = batched_input
        self.angle_weights = angle_weights
        self.is_weighted_loss = True

    def _compute_angle_weights(self, pos, edge_index, edge_pairs) -> torch.Tensor:
        """
        Compute angle-based weights for edge pairs.
        Let u, v be edge vectors:
        $w_{ij} = |cos(\theta_{uv})|$ helps the loss understand "straight" anatomical lines vs. "sharp" turns.

        Args:
            pos: [V, 3] vertex positions
            edge_index: [2, E] edge connectivity
            edge_pairs: [2, P] adjacent edge pairs
        """
        e1_idx = edge_pairs[0]  # [P]
        e2_idx = edge_pairs[1]  # [P]

        # Get edge vectors
        edge_vecs = pos[edge_index[1]] - pos[edge_index[0]]  # [E, 3]

        u = edge_vecs[e1_idx]  # [P, 3]
        v = edge_vecs[e2_idx]  # [P, 3]

        # Normalize vectors
        u_norm = F.normalize(u, p=2, dim=1)  # [P, 3]
        v_norm = F.normalize(v, p=2, dim=1)  # [P, 3]

        # Compute cosine of angle between u and v
        cos_theta = torch.abs(torch.sum(u_norm * v_norm, dim=1))  # [P]

        return cos_theta  # [P]

    @property
    def required_attributes(self) -> List[str]:
        return ["pos", "edge_index"]

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        w_loss: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute boundary multi-loss.

        Args:
            predictions: [E, C] unnormalized logits
            targets: [E] ground truth labels (0, 1, 2, or -1 to ignore)
            w_loss: [E, C] edge weights per class
            data: PyTorch Geometric Data/Batch object containing edge_index

        Returns:
            Scalar loss value
        """
        # Check kwargs has pos and edge_index
        assert all(attr in kwargs for attr in self.required_attributes)
        if w_loss is None:
            w_loss = torch.ones_like(predictions, dtype=torch.float32)

        pos = kwargs["pos"]
        edge_index = kwargs["edge_index"]

        self.label_weights = (
            self.label_weights.to(predictions.device)
            if self.label_weights is not None
            else None
        )
        loss = self.ce_scalar * F.cross_entropy(
            predictions,
            targets,
            weight=self.label_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,
        )

        # Handle ignored labels (-1)
        valid_mask = targets != -1

        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device)

        # Filter to valid edges only
        valid_predictions = predictions[valid_mask]  # [E_valid, C]
        valid_targets = targets[valid_mask]  # [E_valid]
        valid_edge_index = edge_index[:, valid_mask]  # [2, E_valid]

        # Get adjacent edge pairs with different labels
        edge_pairs = get_adjacent_edge_pairs(
            edges=valid_edge_index,
            num_vertices=pos.shape[0],
            edge_labels=valid_targets,
        )

        if edge_pairs.shape[1] == 0:
            # No valid edge pairs found
            return torch.tensor(0.0, device=predictions.device)

        e1_idx = edge_pairs[0]  # [P] indices of first edges
        e2_idx = edge_pairs[1]  # [P] indices of second edges

        # Apply softmax to get probability distributions
        probs = F.softmax(valid_predictions, dim=1)  # [E_valid, C]

        # Compute L1 distances between probability distributions
        # ||p_e1 - p_e2||_1 = sum_c |p_e1[c] - p_e2[c]|
        l1_distances = torch.sum(torch.abs(probs[e1_idx] - probs[e2_idx]), dim=1)  # [P]

        # Compute weights for each edge pair (mean of per-edge weights)
        # w_loss is [E, C], take mean across classes for each edge
        if self.angle_weights:
            w_pairs = self._compute_angle_weights(
                pos, valid_edge_index, edge_pairs
            )  # [P]
        else:
            edge_weights = w_loss.mean(dim=1)  # [E]
            valid_edge_weights = edge_weights[valid_mask]  # [E_valid]
            w_pairs = (
                valid_edge_weights[e1_idx] + valid_edge_weights[e2_idx]
            ) / 2.0  # [P]

        # Compute boundary term: w_uv * (1 - 0.5 * ||p_e1 - p_e2||_1)
        boundary_terms = w_pairs * (1.0 - 0.5 * l1_distances)  # [P]
        # Sum over all pairs and normalize by total weight
        loss += boundary_terms.sum() / (w_pairs.sum() + 1e-8)

        return loss


class EdgeLocalContinuityLoss(BaseLoss):
    """Penalize disagreement between adjacent edge prediction distributions.

    This is a standalone regularizer intended to complement the supervised
    prediction loss, not replace it.
    """

    def __init__(
        self,
        distance: str = "tv",
        eps: float = 1e-8,
        **kwargs: Any,
    ):
        if distance not in {"tv", "js"}:
            raise ValueError(
                f"Unsupported distance '{distance}'. Expected one of ['tv', 'js']."
            )
        self.distance = distance
        self.eps = eps
        self.is_weighted_loss = False

    @property
    def required_attributes(self) -> List[str]:
        return ["edge_index"]

    @property
    def model_output_key(self) -> str | None:
        return "pred"

    def _pairwise_distance(self, probs: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        first_idx = edge_pairs[0]
        second_idx = edge_pairs[1]

        first_probs = probs[first_idx]
        second_probs = probs[second_idx]

        if self.distance == "tv":
            return 0.5 * torch.sum(torch.abs(first_probs - second_probs), dim=1)

        midpoint = 0.5 * (first_probs + second_probs)
        kl_first = torch.sum(
            first_probs * (torch.log(first_probs + self.eps) - torch.log(midpoint + self.eps)),
            dim=1,
        )
        kl_second = torch.sum(
            second_probs * (torch.log(second_probs + self.eps) - torch.log(midpoint + self.eps)),
            dim=1,
        )
        return 0.5 * (kl_first + kl_second)

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert all(attr in kwargs for attr in self.required_attributes)
        edge_index = kwargs["edge_index"]

        if predictions.numel() == 0 or edge_index.numel() == 0:
            return predictions.new_tensor(0.0)

        valid_mask = torch.ones(
            predictions.shape[0], dtype=torch.bool, device=predictions.device
        )
        if targets is not None:
            valid_mask = targets != -1
            if not valid_mask.any():
                return predictions.new_tensor(0.0)

        valid_predictions = predictions[valid_mask]
        valid_edge_index = edge_index[:, valid_mask]
        num_vertices = int(valid_edge_index.max().item()) + 1

        edge_pairs = get_adjacent_edge_pairs(
            edges=valid_edge_index,
            num_vertices=num_vertices,
            edge_labels=None,
        )
        if edge_pairs.shape[1] == 0:
            return predictions.new_tensor(0.0)

        probs = F.softmax(valid_predictions, dim=1)
        pairwise_distance = self._pairwise_distance(probs, edge_pairs)
        return pairwise_distance.mean()


class BoundaryDoULoss(BaseLoss):
    """
    Boundary-aware Degree of Union (DoU) Loss for mesh edge segmentation.

    Adapted from 2D image segmentation to graph-based mesh edges. Uses edge adjacency
    to detect boundaries and adaptively weights the intersection based on boundary proportion.

    Loss formula per class: L = (G ∪ P - G ∩ P) / (G ∪ P - α * G ∩ P)
    where α = min(2 * (1 - C/S) - 1, 0.8), C = boundary edges, S = foreground edges

    Reference: Boundary DoU loss for improved segmentation of thin structures

    Args:
        n_classes: Number of segmentation classes
        alpha_max: Maximum value for alpha truncation (default: 0.8)
        smooth: Smoothing constant for numerical stability (default: 1e-5)
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_weights: List[float] = None,
        ce_scalar: float = 1.0,
        alpha_max: float = 0.8,
        smooth: float = 1e-6,
        batched_input: bool = True,
    ):
        self.label_smoothing = label_smoothing
        self.label_weights = (
            torch.tensor(label_weights) if label_weights is not None else None
        )
        self.ce_scalar = ce_scalar
        self.alpha_max = alpha_max
        self.smooth = smooth
        self.is_weighted_loss = False
        self.batched_input = batched_input

    @property
    def required_attributes(self) -> List[str]:
        return ["edge_index", "pos"]

    def _compute_boundary_mask(
        self,
        edge_index: torch.Tensor,
        num_vertices: int,
        targets: torch.Tensor,
        class_idx: int,
    ) -> torch.Tensor:
        """
        Compute binary mask indicating boundary edges for a specific class.
        A boundary edge is one that has at least one adjacent edge with a different label.

        Args:
            edge_index: [2, E] edge connectivity
            num_vertices: Number of vertices
            targets: [E] edge labels
            class_idx: Class index to compute boundaries for

        Returns:
            boundary_mask: [E] binary mask (1 = boundary edge, 0 = interior edge)
        """
        # Get all adjacent edge pairs (no label filtering)
        edge_pairs = get_adjacent_edge_pairs(
            edges=edge_index,
            num_vertices=num_vertices,
            edge_labels=None,  # Don't filter by label
        )

        if edge_pairs.shape[1] == 0:
            return torch.zeros(
                targets.shape[0], dtype=torch.bool, device=targets.device
            )

        e1_idx = edge_pairs[0]
        e2_idx = edge_pairs[1]

        # Find edges of current class that are adjacent to edges of different class
        is_class_e1 = targets[e1_idx] == class_idx
        is_class_e2 = targets[e2_idx] == class_idx
        diff_label = targets[e1_idx] != targets[e2_idx]

        # Mark e1 as boundary if it's the target class and e2 is different
        boundary_e1 = is_class_e1 & diff_label
        # Mark e2 as boundary if it's the target class and e1 is different
        boundary_e2 = is_class_e2 & diff_label

        # Create boundary mask
        boundary_mask = torch.zeros(
            targets.shape[0], dtype=torch.bool, device=targets.device
        )
        boundary_mask[e1_idx[boundary_e1]] = True
        boundary_mask[e2_idx[boundary_e2]] = True

        return boundary_mask

    def _adaptive_size_mesh(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive DoU loss for one class.

        Args:
            score: [E] predicted probabilities for this class
            target: [E] binary ground truth for this class
            boundary_mask: [E] binary mask indicating boundary edges

        Returns:
            Scalar loss value
        """
        # Count boundary edges (C) and foreground edges (S)
        C = torch.count_nonzero(boundary_mask * target).float()
        S = torch.count_nonzero(target).float()

        # Compute adaptive alpha
        alpha = 1.0 - (C + self.smooth) / (S + self.smooth)
        alpha = 2.0 * alpha - 1.0
        alpha = min(alpha.item(), self.alpha_max)  # Truncate alpha

        # Compute DoU loss: L = (|G ∪ P| - |G ∩ P|) / (|G ∪ P| - α|G ∩ P|)
        # |G ∪ P| = |G| + |P| - |G ∩ P|
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)  # = torch.sum(target) for binary
        z_sum = torch.sum(score * score)  # For probabilities

        # Numerator: |G| + |P| - 2|G ∩ P|
        numerator = z_sum + y_sum - 2.0 * intersect + self.smooth

        # Denominator: |G| + |P| - (1+α)|G ∩ P|
        denominator = z_sum + y_sum - (1.0 + alpha) * intersect + self.smooth

        loss = numerator / denominator

        return loss

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute Boundary DoU loss.

        Args:
            predictions: [E, C] unnormalized logits
            targets: [E] ground truth labels (0, 1, 2, ..., or -1 to ignore)
            kwargs: Must contain 'edge_index' and 'pos'

        Returns:
            Scalar loss value averaged across classes
        """
        # Check required attributes
        assert all(
            attr in kwargs for attr in self.required_attributes
        ), f"Missing required attributes: {self.required_attributes}"

        edge_index = kwargs["edge_index"]
        pos = kwargs["pos"]
        num_vertices = pos.shape[0]
        n_classes = predictions.shape[-1]

        self.label_weights = (
            self.label_weights.to(predictions.device)
            if self.label_weights is not None
            else None
        )

        loss = self.ce_scalar * F.cross_entropy(
            predictions,
            targets,
            weight=self.label_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,
        )

        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)  # [E, C]

        # Handle ignored labels (-1)
        valid_mask = targets != -1

        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device)

        # Filter to valid edges only
        valid_probs = probs[valid_mask]  # [E_valid, C]
        valid_targets = targets[valid_mask]  # [E_valid]
        valid_edge_index = edge_index[:, valid_mask]  # [2, E_valid]

        bdou_total_loss = 0.0

        for class_idx in range(n_classes):
            # Binary ground truth for this class
            target_binary = (valid_targets == class_idx).float()  # [E_valid]

            if target_binary.sum() == 0:
                # Skip classes with no ground truth samples
                continue

            # Get predictions for this class
            score = valid_probs[:, class_idx]  # [E_valid]

            # Compute boundary mask
            boundary_mask = self._compute_boundary_mask(
                edge_index=valid_edge_index,
                num_vertices=num_vertices,
                targets=valid_targets,
                class_idx=class_idx,
            )

            # Compute per-class loss
            class_loss = self._adaptive_size_mesh(
                score=score,
                target=target_binary,
                boundary_mask=boundary_mask,
            )

            bdou_total_loss += class_loss

        # Average across classes
        return loss + bdou_total_loss / n_classes


###### Vertex-wise Losses ######


def get_adjacent_vertex_pairs(
    edge_index: torch.Tensor, vertex_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return undirected adjacent vertex pairs from `edge_index`.

    Args:
        edge_index: Tensor of shape (2, E)
        vertex_labels: Optional tensor of shape (N,) to filter boundary pairs

    Returns:
        pairs: LongTensor of shape (2, P) with u < v.

    Notes:
        - Self-loops are removed.
        - Directed duplicates are coalesced to a single undirected pair.
        - If `vertex_labels` is provided, returns only pairs where labels differ
          and both endpoints are valid (label != -1).
    """

    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    edges = edge_index.t().long()  # (E, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    edges_sorted, _ = torch.sort(edges, dim=1)
    edges_unique = torch.unique(edges_sorted, dim=0)
    u = edges_unique[:, 0]
    v = edges_unique[:, 1]

    if vertex_labels is not None:
        vertex_labels = vertex_labels.long()
        valid = (vertex_labels[u] != -1) & (vertex_labels[v] != -1)
        diff = vertex_labels[u] != vertex_labels[v]
        keep = valid & diff
        u = u[keep]
        v = v[keep]

    return torch.stack([u, v], dim=0)


def get_adjacent_face_pairs(
    face_adjacency: torch.Tensor, face_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return undirected adjacent face pairs from `face_adjacency`.

    Args:
        face_adjacency: Tensor of shape (P, 2) or (2, P)
        face_labels: Optional tensor of shape (F,) to filter boundary pairs

    Returns:
        pairs: LongTensor of shape (2, P) with u < v.
    """

    if face_adjacency.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=face_adjacency.device)

    pairs = face_adjacency.long()
    if pairs.ndim != 2:
        raise ValueError(
            f"Expected face_adjacency to be 2D, got shape={tuple(pairs.shape)}"
        )
    if pairs.shape[0] == 2 and pairs.shape[1] != 2:
        pairs = pairs.t().contiguous()
    elif pairs.shape[1] != 2:
        raise ValueError(
            f"Expected face_adjacency shape (P,2) or (2,P), got {tuple(pairs.shape)}"
        )

    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=face_adjacency.device)

    pairs_sorted, _ = torch.sort(pairs, dim=1)
    pairs_unique = torch.unique(pairs_sorted, dim=0)
    u = pairs_unique[:, 0]
    v = pairs_unique[:, 1]

    if face_labels is not None:
        face_labels = face_labels.long()
        valid = (face_labels[u] != -1) & (face_labels[v] != -1)
        diff = face_labels[u] != face_labels[v]
        keep = valid & diff
        u = u[keep]
        v = v[keep]

    return torch.stack([u, v], dim=0)


class VertexGraphCutLoss(BaseLoss):
    """Vertex-wise version of GraphCutLoss.

    Penalizes prediction disagreement between adjacent vertices with different
    ground-truth labels.

    Uses adjacency induced by `edge_index` (vertex graph).
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_weights: List[float] = None,
        ce_scalar: float = 1.0,
        batched_input: bool = True,
    ):
        self.label_smoothing = label_smoothing
        self.label_weights = (
            torch.tensor(label_weights) if label_weights is not None else None
        )
        self.ce_scalar = ce_scalar
        self.batched_input = batched_input
        self.is_weighted_loss = True

    @property
    def required_attributes(self) -> List[str]:
        return ["edge_index"]

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        w_loss: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert all(attr in kwargs for attr in self.required_attributes)
        edge_index = kwargs["edge_index"]

        if w_loss is None:
            w_loss = torch.ones_like(predictions, dtype=torch.float32)

        self.label_weights = (
            self.label_weights.to(predictions.device)
            if self.label_weights is not None
            else None
        )

        ce = self.ce_scalar * F.cross_entropy(
            predictions,
            targets,
            weight=self.label_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,
        )

        valid_nodes = targets != -1
        if not valid_nodes.any():
            return torch.tensor(0.0, device=predictions.device)

        pairs = get_adjacent_vertex_pairs(edge_index=edge_index, vertex_labels=targets)
        if pairs.shape[1] == 0:
            return ce

        u, v = pairs[0], pairs[1]
        probs = F.softmax(predictions, dim=1)
        l1 = torch.sum(torch.abs(probs[u] - probs[v]), dim=1)  # (P,)

        node_w = w_loss.mean(dim=1)  # (N,)
        w_pairs = 0.5 * (node_w[u] + node_w[v])  # (P,)
        boundary_terms = w_pairs * (1.0 - 0.5 * l1)
        smooth_term = boundary_terms.sum() / (w_pairs.sum() + 1e-8)

        return ce + smooth_term


class VertexBoundaryDoULoss(BaseLoss):
    """Vertex-wise Boundary DoU loss.

    Adapts BoundaryDoULoss to vertex segmentation with adjacency defined by `edge_index`.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_weights: List[float] = None,
        ce_scalar: float = 1.0,
        alpha_max: float = 0.8,
        smooth: float = 1e-6,
        batched_input: bool = True,
    ):
        self.label_smoothing = label_smoothing
        self.label_weights = (
            torch.tensor(label_weights) if label_weights is not None else None
        )
        self.ce_scalar = ce_scalar
        self.alpha_max = alpha_max
        self.smooth = smooth
        self.batched_input = batched_input
        self.is_weighted_loss = False

    @property
    def required_attributes(self) -> List[str]:
        return ["edge_index"]

    def _compute_boundary_mask(
        self, edge_index: torch.Tensor, targets: torch.Tensor, class_idx: int
    ) -> torch.Tensor:
        """Boundary vertices are class vertices adjacent to a different class."""

        n = targets.shape[0]
        boundary = torch.zeros(n, dtype=torch.bool, device=targets.device)

        pairs = get_adjacent_vertex_pairs(edge_index=edge_index)
        if pairs.shape[1] == 0:
            return boundary

        u, v = pairs[0], pairs[1]
        tu = targets[u]
        tv = targets[v]

        valid = (tu != -1) & (tv != -1)
        diff = tu != tv

        # u is boundary if u is class_idx and v differs
        mask_u = valid & diff & (tu == class_idx)
        mask_v = valid & diff & (tv == class_idx)
        boundary[u[mask_u]] = True
        boundary[v[mask_v]] = True
        return boundary

    def _adaptive_size_vertex(
        self, score: torch.Tensor, target: torch.Tensor, boundary_mask: torch.Tensor
    ) -> torch.Tensor:
        C = torch.count_nonzero(boundary_mask * target.bool()).float()
        S = torch.count_nonzero(target.bool()).float()

        alpha = 1.0 - (C + self.smooth) / (S + self.smooth)
        alpha = 2.0 * alpha - 1.0
        alpha = min(alpha.item(), self.alpha_max)

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        numerator = z_sum + y_sum - 2.0 * intersect + self.smooth
        denominator = z_sum + y_sum - (1.0 + alpha) * intersect + self.smooth
        return numerator / denominator

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert all(attr in kwargs for attr in self.required_attributes)
        edge_index = kwargs["edge_index"]
        n_classes = predictions.shape[-1]

        self.label_weights = (
            self.label_weights.to(predictions.device)
            if self.label_weights is not None
            else None
        )

        ce = self.ce_scalar * F.cross_entropy(
            predictions,
            targets,
            weight=self.label_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,
        )

        probs = F.softmax(predictions, dim=1)
        valid_mask = targets != -1
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device)

        bdou_total = 0.0
        for class_idx in range(n_classes):
            target_binary = (targets == class_idx).float()
            target_binary = target_binary * valid_mask.float()
            if target_binary.sum() == 0:
                continue

            score = probs[:, class_idx]
            boundary_mask = self._compute_boundary_mask(edge_index, targets, class_idx)
            class_loss = self._adaptive_size_vertex(score, target_binary, boundary_mask)
            bdou_total += class_loss

        return ce + bdou_total / max(n_classes, 1)


class FaceGraphCutLoss(BaseLoss):
    """Face-wise version of GraphCutLoss using face adjacency."""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_weights: List[float] = None,
        ce_scalar: float = 1.0,
        batched_input: bool = True,
    ):
        self.label_smoothing = label_smoothing
        self.label_weights = (
            torch.tensor(label_weights) if label_weights is not None else None
        )
        self.ce_scalar = ce_scalar
        self.batched_input = batched_input
        self.is_weighted_loss = True

    @property
    def required_attributes(self) -> List[str]:
        return ["face_adjacency"]

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        w_loss: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert all(attr in kwargs for attr in self.required_attributes)
        face_adjacency = kwargs["face_adjacency"]

        if w_loss is None:
            w_loss = torch.ones_like(predictions, dtype=torch.float32)

        self.label_weights = (
            self.label_weights.to(predictions.device)
            if self.label_weights is not None
            else None
        )

        ce = self.ce_scalar * F.cross_entropy(
            predictions,
            targets,
            weight=self.label_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,
        )

        valid_faces = targets != -1
        if not valid_faces.any():
            return torch.tensor(0.0, device=predictions.device)

        pairs = get_adjacent_face_pairs(
            face_adjacency=face_adjacency,
            face_labels=targets,
        )
        if pairs.shape[1] == 0:
            return ce

        u, v = pairs[0], pairs[1]
        probs = F.softmax(predictions, dim=1)
        l1 = torch.sum(torch.abs(probs[u] - probs[v]), dim=1)

        face_w = w_loss.mean(dim=1)
        w_pairs = 0.5 * (face_w[u] + face_w[v])
        boundary_terms = w_pairs * (1.0 - 0.5 * l1)
        smooth_term = boundary_terms.sum() / (w_pairs.sum() + 1e-8)
        return ce + smooth_term


class FaceBoundaryDoULoss(BaseLoss):
    """Face-wise Boundary DoU loss using face adjacency."""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        label_weights: List[float] = None,
        ce_scalar: float = 1.0,
        alpha_max: float = 0.8,
        smooth: float = 1e-6,
        batched_input: bool = True,
    ):
        self.label_smoothing = label_smoothing
        self.label_weights = (
            torch.tensor(label_weights) if label_weights is not None else None
        )
        self.ce_scalar = ce_scalar
        self.alpha_max = alpha_max
        self.smooth = smooth
        self.batched_input = batched_input
        self.is_weighted_loss = False

    @property
    def required_attributes(self) -> List[str]:
        return ["face_adjacency"]

    def _compute_boundary_mask(
        self,
        face_adjacency: torch.Tensor,
        targets: torch.Tensor,
        class_idx: int,
    ) -> torch.Tensor:
        boundary = torch.zeros(
            targets.shape[0], dtype=torch.bool, device=targets.device
        )

        pairs = get_adjacent_face_pairs(face_adjacency=face_adjacency)
        if pairs.shape[1] == 0:
            return boundary

        u, v = pairs[0], pairs[1]
        tu = targets[u]
        tv = targets[v]

        valid = (tu != -1) & (tv != -1)
        diff = tu != tv
        mask_u = valid & diff & (tu == class_idx)
        mask_v = valid & diff & (tv == class_idx)
        boundary[u[mask_u]] = True
        boundary[v[mask_v]] = True
        return boundary

    def _adaptive_size_face(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        C = torch.count_nonzero(boundary_mask * target.bool()).float()
        S = torch.count_nonzero(target.bool()).float()

        alpha = 1.0 - (C + self.smooth) / (S + self.smooth)
        alpha = 2.0 * alpha - 1.0
        alpha = min(alpha.item(), self.alpha_max)

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        numerator = z_sum + y_sum - 2.0 * intersect + self.smooth
        denominator = z_sum + y_sum - (1.0 + alpha) * intersect + self.smooth
        return numerator / denominator

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert all(attr in kwargs for attr in self.required_attributes)
        face_adjacency = kwargs["face_adjacency"]
        n_classes = predictions.shape[-1]

        self.label_weights = (
            self.label_weights.to(predictions.device)
            if self.label_weights is not None
            else None
        )

        ce = self.ce_scalar * F.cross_entropy(
            predictions,
            targets,
            weight=self.label_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,
        )

        probs = F.softmax(predictions, dim=1)
        valid_mask = targets != -1
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device)

        bdou_total = 0.0
        for class_idx in range(n_classes):
            target_binary = (targets == class_idx).float()
            target_binary = target_binary * valid_mask.float()
            if target_binary.sum() == 0:
                continue

            score = probs[:, class_idx]
            boundary_mask = self._compute_boundary_mask(
                face_adjacency=face_adjacency,
                targets=targets,
                class_idx=class_idx,
            )
            bdou_total += self._adaptive_size_face(
                score=score,
                target=target_binary,
                boundary_mask=boundary_mask,
            )

        return ce + bdou_total / max(n_classes, 1)


class ContrastiveBoundaryLoss(BaseLoss):
    """Radius-neighborhood contrastive boundary loss.

    Operates on per-vertex embeddings `predictions` and vertex labels `targets`.

    Implements:
        - 1/|P| * sum_{x in P} log( sum_{y in N_x, Lx=Ly} exp(-d(Fx,Fy)) / sum_{y in N_x} exp(-d(Fx,Fy)) )

    Where N_x are neighbors within radius `r`.
    """

    def __init__(
        self,
        k: int = 8,
        eps: float = 1e-8,
        batched_input: bool = True,
    ):
        self.k = int(k)
        self.eps = float(eps)
        self.batched_input = batched_input
        self.is_weighted_loss = False

    @property
    def supervision_key(self) -> str | None:
        return "pred"

    @property
    def required_attributes(self) -> List[str]:
        # User requested only pos required; will optionally use batch if provided.
        return ["pos"]

    def __call__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert "pos" in kwargs
        if labels is None:
            raise ValueError("ContrastiveBoundaryLoss requires label supervision")

        pos: torch.Tensor = kwargs["pos"]
        batch = kwargs.get("batch", None)

        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)

        # Valid points set P
        valid = labels != -1
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Build KNN-based neighbor pairs (row = query index, col = neighbor index)
        try:
            from torch_geometric.nn import knn
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ContrastiveBoundaryLoss requires torch_geometric (knn op)."
            ) from e

        row, col = knn(
            pos,
            pos,
            k=self.k,
            batch_x=batch,
            batch_y=batch,
        )
        # Remove self-neighbors
        mask = row != col
        row = row[mask]
        col = col[mask]

        # Remove edges involving invalid labels
        mask = valid[row] & valid[col]
        row = row[mask]
        col = col[mask]

        if row.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute exp(-||Fx - Fy||_2)
        fx = embeddings[row]
        fy = embeddings[col]
        dist = torch.norm(fx - fy, dim=1)  # (E,)
        w = torch.exp(-dist)

        same = (labels[row] == labels[col]).to(w.dtype)
        w_same = w * same

        # Aggregate numerator/denominator per query vertex
        try:
            from torch_scatter import scatter
        except Exception as e:  # pragma: no cover
            raise ImportError("ContrastiveBoundaryLoss requires torch_scatter.") from e

        denom = scatter(w, row, dim=0, dim_size=pos.shape[0], reduce="sum")
        numer = scatter(w_same, row, dim=0, dim_size=pos.shape[0], reduce="sum")

        has_nbrs = denom > 0
        use = valid & has_nbrs
        if not use.any():
            return torch.tensor(0.0, device=embeddings.device)

        p = numer[use] / (denom[use] + self.eps)
        loss = -torch.mean(torch.log(p + self.eps))
        return loss


class RegionalAssignmentLoss(BaseLoss):
    def __init__(
        self,
        diversity_weight: float = 1.0,
        equipartition_weight: float = 1.0,
        eps: float = 1e-8,
        batched_input: bool = True,
    ):
        self.diversity_weight = float(diversity_weight)
        self.equipartition_weight = float(equipartition_weight)
        self.eps = float(eps)
        self.batched_input = batched_input
        self.is_weighted_loss = False
        self.requires_supervision = False

    @property
    def required_attributes(self) -> List[str]:
        return ["batch"]

    def _diversity_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(assignment, p=2, dim=0, eps=self.eps)
        gram = normalized.transpose(0, 1) @ normalized
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return torch.mean((gram - eye) ** 2)

    def _equipartition_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        mass = assignment.sum(dim=0)
        mean_mass = mass.mean()
        return torch.mean((mass - mean_mass) ** 2) / (mean_mass.square() + self.eps)

    def __call__(
        self,
        assignment_matrix: torch.Tensor,
        supervision: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del supervision

        if "batch" not in kwargs:
            raise ValueError("RegionalAssignmentLoss requires `batch`")

        batch: torch.Tensor = kwargs["batch"]
        if assignment_matrix.numel() == 0 or batch.numel() == 0:
            return torch.tensor(0.0, device=assignment_matrix.device)

        num_graphs = int(batch.max().item()) + 1
        total_loss = assignment_matrix.new_tensor(0.0)
        num_valid_graphs = 0

        for graph_idx in range(num_graphs):
            graph_mask = batch == graph_idx
            if not graph_mask.any():
                continue

            graph_assignment = assignment_matrix[graph_mask]
            diversity = self._diversity_loss(graph_assignment)
            equipartition = self._equipartition_loss(graph_assignment)
            total_loss = total_loss + (
                self.diversity_weight * diversity
                + self.equipartition_weight * equipartition
            )
            num_valid_graphs += 1

        if num_valid_graphs == 0:
            return assignment_matrix.new_tensor(0.0)

        return total_loss / num_valid_graphs


class VirtualNodeMMDLoss(BaseLoss):
    def __init__(
        self,
        sigma: float = 1.0,
        sample_size: int = 256,
        vv_weight: float = 1.0,
        rv_weight: float = 2.0,
        eps: float = 1e-8,
        batched_input: bool = True,
    ):
        self.sigma = float(sigma)
        self.sample_size = int(sample_size)
        self.vv_weight = float(vv_weight)
        self.rv_weight = float(rv_weight)
        self.eps = float(eps)
        self.batched_input = batched_input
        self.is_weighted_loss = False
        self.requires_supervision = False

        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got: {self.sigma}")

    @property
    def required_attributes(self) -> List[str]:
        return ["batch", "pos"]

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sq_dist = torch.cdist(x, y, p=2).square()
        return torch.exp(-sq_dist / (2.0 * (self.sigma**2) + self.eps))

    def _sample_real_nodes(self, pos: torch.Tensor) -> torch.Tensor:
        if self.sample_size <= 0 or pos.shape[0] <= self.sample_size:
            return pos
        keep = torch.randperm(pos.shape[0], device=pos.device)[: self.sample_size]
        return pos[keep]

    def _geometry_positions(self, virtual_node_coords: torch.Tensor) -> torch.Tensor:
        if virtual_node_coords.ndim == 3 and virtual_node_coords.shape[-1] == 3:
            return virtual_node_coords
        if virtual_node_coords.ndim == 4 and virtual_node_coords.shape[2] == 3:
            return virtual_node_coords.mean(dim=-1)
        raise ValueError(
            "virtual_node_coords must have shape (B, V, 3) or (B, V, 3, C), got: "
            f"{tuple(virtual_node_coords.shape)}"
        )

    def __call__(
        self,
        virtual_node_coords: torch.Tensor,
        supervision: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del supervision

        batch = kwargs["batch"]
        pos = kwargs["pos"]

        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got: {tuple(pos.shape)}")
        if batch.numel() == 0 or pos.numel() == 0 or virtual_node_coords.numel() == 0:
            return pos.new_tensor(0.0)

        virtual_node_coords = self._geometry_positions(virtual_node_coords)

        num_graphs = int(batch.max().item()) + 1
        if virtual_node_coords.shape[0] != num_graphs:
            raise ValueError(
                "virtual_node_coords batch dimension must match number of graphs, got: "
                f"{virtual_node_coords.shape[0]} vs {num_graphs}"
            )

        total_loss = pos.new_tensor(0.0)
        valid_graphs = 0
        for graph_idx in range(num_graphs):
            graph_mask = batch == graph_idx
            if not graph_mask.any():
                continue

            real_pos = self._sample_real_nodes(pos[graph_mask])
            virtual_pos = virtual_node_coords[graph_idx]
            if real_pos.numel() == 0 or virtual_pos.numel() == 0:
                continue

            k_rv = self._gaussian_kernel(real_pos, virtual_pos).mean()
            k_vv = self._gaussian_kernel(virtual_pos, virtual_pos)
            if k_vv.shape[0] > 1:
                diag_mask = ~torch.eye(
                    k_vv.shape[0], device=k_vv.device, dtype=torch.bool
                )
                k_vv = k_vv[diag_mask].mean()
            else:
                k_vv = k_vv.new_tensor(0.0)

            total_loss = total_loss + self.vv_weight * k_vv - self.rv_weight * k_rv
            valid_graphs += 1

        if valid_graphs == 0:
            return pos.new_tensor(0.0)

        return total_loss / valid_graphs


class LatentCoordinateRegularizationLoss(BaseLoss):
    def __init__(
        self,
        displacement_weight: float = 1.0,
        edge_distortion_weight: float = 1.0,
        face_area_weight: float = 1.0,
        eps: float = 1e-8,
        batched_input: bool = True,
    ):
        self.displacement_weight = float(displacement_weight)
        self.edge_distortion_weight = float(edge_distortion_weight)
        self.face_area_weight = float(face_area_weight)
        self.eps = float(eps)
        self.batched_input = batched_input
        self.is_weighted_loss = False
        self.requires_supervision = False

    @property
    def model_output_key(self) -> str | None:
        return "latent_pos"

    @property
    def required_attributes(self) -> List[str]:
        return ["batch", "pos", "edge_index", "face"]

    def _face_areas(self, coord: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        if face.numel() == 0:
            return coord.new_empty((0,))
        i, j, k = face[0].long(), face[1].long(), face[2].long()
        normal = torch.cross(coord[j] - coord[i], coord[k] - coord[j], dim=1)
        return normal.norm(dim=-1).clamp_min(self.eps) / 2.0

    def __call__(
        self,
        latent_pos: torch.Tensor,
        supervision: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del supervision

        batch = kwargs["batch"].long()
        pos = kwargs["pos"]
        edge_index = kwargs["edge_index"].long()
        face = kwargs["face"].long()

        if latent_pos.ndim != 2 or latent_pos.shape[-1] != 3:
            raise ValueError(
                "latent_pos must have shape (N, 3), got: "
                f"{tuple(latent_pos.shape)}"
            )
        if pos.shape != latent_pos.shape:
            raise ValueError(
                "latent_pos and pos must have matching shapes, got: "
                f"{tuple(latent_pos.shape)} and {tuple(pos.shape)}"
            )
        if batch.shape[0] != pos.shape[0]:
            raise ValueError(
                "batch must align with pos/latent_pos, got: "
                f"{tuple(batch.shape)} vs {tuple(pos.shape)}"
            )

        if pos.numel() == 0:
            return pos.new_tensor(0.0)

        displacement = latent_pos - pos
        edge_batch = (
            batch[edge_index[0].long()]
            if edge_index.numel() > 0
            else batch.new_empty((0,), dtype=batch.dtype)
        )
        face_batch = (
            batch[face[0].long()]
            if face.numel() > 0
            else batch.new_empty((0,), dtype=batch.dtype)
        )

        num_graphs = int(batch.max().item()) + 1
        total_loss = pos.new_tensor(0.0)
        valid_graphs = 0

        for graph_idx in range(num_graphs):
            node_mask = batch == graph_idx
            if not node_mask.any():
                continue

            graph_loss = pos.new_tensor(0.0)

            if self.displacement_weight != 0.0:
                graph_displacement = displacement[node_mask]
                graph_loss = graph_loss + self.displacement_weight * (
                    graph_displacement.square().sum(dim=-1).mean()
                )

            if self.edge_distortion_weight != 0.0 and edge_index.numel() > 0:
                graph_edge_mask = edge_batch == graph_idx
                if graph_edge_mask.any():
                    src = edge_index[0, graph_edge_mask].long()
                    dst = edge_index[1, graph_edge_mask].long()
                    base_len = torch.norm(pos[src] - pos[dst], dim=-1)
                    latent_len = torch.norm(latent_pos[src] - latent_pos[dst], dim=-1)
                    graph_loss = graph_loss + self.edge_distortion_weight * (
                        (latent_len - base_len).square().mean()
                    )

            if self.face_area_weight != 0.0 and face.numel() > 0:
                graph_face_mask = face_batch == graph_idx
                if graph_face_mask.any():
                    graph_face = face[:, graph_face_mask]
                    base_area = self._face_areas(pos, graph_face)
                    latent_area = self._face_areas(latent_pos, graph_face)
                    graph_loss = graph_loss + self.face_area_weight * (
                        (latent_area - base_area).square().mean()
                    )

            total_loss = total_loss + graph_loss
            valid_graphs += 1

        if valid_graphs == 0:
            return pos.new_tensor(0.0)

        return total_loss / valid_graphs


###### Loss Functions ######


def get_adjacent_edge_pairs(edges, num_vertices, edge_labels=None):
    """
    edges: Tensor of shape (2, num_edges)
    returns: Tensor of shape (2, num_adjacent_pairs)
    """
    edges = edges.t()  # (num_edges, 2)
    num_edges = edges.shape[0]

    # 1. Build the sparse incidence matrix H (|V| x |E|)
    # Every edge e=(u,v) has entries at (u, e) and (v, e)
    edge_indices = torch.arange(num_edges, device=edges.device)

    # We flatten the edges to get vertex indices and repeat edge indices
    v_idx = edges.flatten()
    e_idx = edge_indices.repeat_interleave(2)

    indices = torch.stack([v_idx, e_idx])
    values = torch.ones(indices.shape[1], device=edges.device)

    H = torch.sparse_coo_tensor(indices, values, (num_vertices, num_edges))

    # 2. Compute H^T * H to find edge adjacencies
    # This gives an |E| x |E| matrix where entry (i,j) > 0 if edges share a vertex
    A_edge = torch.sparse.mm(H.t(), H)

    # 3. Extract the pairs (i, j) where i < j (to avoid duplicates and self-loops)
    # We only care about entries that are exactly 1 (sharing one vertex)
    A_edge = A_edge.coalesce()
    row, col = A_edge.indices()
    values = A_edge.values()

    mask = (row < col) & (values == 1)  # Only edges sharing exactly 1 vertex

    adj_edge_pairs = torch.stack([row[mask], col[mask]])
    if edge_labels is not None:
        # Filter pairs with different labels
        label_diff_mask = (
            edge_labels[adj_edge_pairs[0]] != edge_labels[adj_edge_pairs[1]]
        )
        adj_edge_pairs = adj_edge_pairs[:, label_diff_mask]
        # Exclude pairs involving -1 labels
        valid_label_mask = (edge_labels[adj_edge_pairs[0]] != -1) & (
            edge_labels[adj_edge_pairs[1]] != -1
        )
        adj_edge_pairs = adj_edge_pairs[:, valid_label_mask]
    return adj_edge_pairs


def dice_coefficient(predictions, targets, smooth=1e-6):
    """
    计算Dice系数。

    参数:
    - predictions: 模型预测结果，维度为[B, N]，B是批次大小，N是点数。
    - targets: 真实标签，维度与predictions相同。
    - smooth: 平滑项，避免除以0的情况，默认为1e-6。
    返回:
    - dice: 计算得到的Dice系数。
    """
    # 确保predictions和targets为二值化数据（0或1）
    predictions = predictions.float()
    targets = targets.float()

    # 计算交集
    intersection = (predictions * targets).sum()

    # 计算Dice系数
    dice = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    # print(dice)
    # 返回批次内的平均Dice系数
    return dice.mean()


def dice_loss_new(output, target, w_loss):
    """
    OPTIMIZED VERSION: Works on flattened PyTorch Geometric batch format.

    Parameters:
    - output: [N, C] where N is total number of elements across batch
    - target: Ground truth labels, size [N], containing 0, 1, 2, and -1 where -1 should be ignored
    - w_loss: Weights, size [N, C] or [C, N]

    Outputs:
    - loss: Dice Loss value
    """
    num_classes = output.shape[1]

    # Ensure w_loss is a tensor on correct device
    if not isinstance(w_loss, torch.Tensor):
        w_loss = torch.from_numpy(w_loss).float().to(output.device)
    elif w_loss.device != output.device or w_loss.dtype != torch.float32:
        w_loss = w_loss.float().to(output.device)

    # Handle different weight shapes - ensure [N, C]
    if w_loss.shape[0] == num_classes and w_loss.shape[1] == output.shape[0]:
        # Weight is [C, N], transpose to [N, C]
        w_loss = w_loss.t()

    # Create valid mask: True where target != -1, shape [N]
    valid_mask = target != -1

    # Filter to valid elements only
    valid_output = output[valid_mask]  # [N_valid, C]
    valid_target = target[valid_mask]  # [N_valid]
    valid_weight = w_loss[valid_mask]  # [N_valid, C]

    # Apply softmax to get probabilities, shape [N_valid, C]
    output_soft = F.softmax(valid_output, dim=1)

    # Create one-hot encoding, shape [N_valid, C]
    target_one_hot = F.one_hot(valid_target.long(), num_classes=num_classes).float()

    # Apply weights: element-wise multiplication
    # Shape: [N_valid, C]
    weighted_output = valid_weight * output_soft
    weighted_target = valid_weight * target_one_hot

    # Compute intersection and union per class
    # Sum over all valid elements (N_valid dimension), result shape: [C]
    intersection = 2 * (weighted_output * target_one_hot).sum(dim=0)  # [C]
    union = weighted_output.sum(dim=0) + weighted_target.sum(dim=0)  # [C]

    # Compute dice per class, shape [C]
    dice_per_class = (intersection + 1e-6) / (union + 1e-6)

    # Dice loss per class
    dice_loss_per_class = 1 - dice_per_class  # [C]

    # Average over classes
    dice_loss_avg = dice_loss_per_class.mean()

    return dice_loss_avg


def dice_loss_new_original(output, target, w_loss):
    """
    ORIGINAL SLOW VERSION - kept for comparison/testing.
    """
    num_classes = output.shape[1]
    batch_size = output.shape[0]
    dice_losses = []

    if not isinstance(w_loss, torch.Tensor):
        w_loss = torch.from_numpy(w_loss).float().to(output.device)
    else:
        w_loss = w_loss.float().to(output.device)

    target_combined = target.clone()

    for i in range(batch_size):
        current_output = output[i]
        current_target = target_combined[i]
        valid_mask = current_target != -1
        valid_output = current_output[:, valid_mask]
        valid_target = current_target[valid_mask]
        weight = w_loss[i][:, valid_mask]

        valid_target_one_hot = (
            F.one_hot(valid_target, num_classes=num_classes).permute(1, 0).float()
        )
        valid_output_soft = F.softmax(valid_output, dim=0)

        intersection = 2 * (weight * valid_output_soft * valid_target_one_hot).sum(-1)
        union = (weight * valid_output_soft).sum(-1) + (
            weight * valid_target_one_hot
        ).sum(-1)
        dice_loss_sample = 1 - (intersection + 1e-6) / (union + 1e-6)
        dice_loss_sample = dice_loss_sample.mean()
        dice_losses.append(dice_loss_sample)

    dice_loss_batch = torch.stack(dice_losses).mean()
    return dice_loss_batch


def dice_loss_standard(output, target, smooth=1e-6):
    """
    计算标准Dice Loss，将前景类别分开计算损失并平均。
    参数:
    - output: 模型输出，大小为[B, 3, H, W]。
    - target: 真实标签，大小为[B, H, W]，包含0, 1, 2和-1，-1需要被忽略。

    返回:
    - loss: 标准Dice Loss的值。
    """
    num_classes = output.shape[1]
    batch_size = output.shape[0]
    dice_losses = []

    for i in range(batch_size):
        # 获取当前样本的输出和标签
        current_output = output[i]  # 大小为[3, H, W]
        current_target = target[i]  # 大小为[H, W]
        # 选择有效的标签点
        valid_mask = current_target != -1  # 大小为[H, W]
        valid_output = current_output[:, valid_mask]  # 大小为[3, valid_HW]
        valid_target = current_target[valid_mask]  # 大小为[valid_HW]

        # 对标签进行one-hot编码，大小为[3, valid_HW]
        valid_target_one_hot = (
            F.one_hot(valid_target, num_classes=num_classes).permute(1, 0).float()
        )
        # 确保输入被softmax处理，转换成概率分布
        valid_output_soft = F.softmax(valid_output, dim=0)

        # 对每个类别分别计算Dice损失
        dice_loss_per_class = []
        for class_idx in range(num_classes):
            intersection = (
                2
                * (valid_output_soft[class_idx] * valid_target_one_hot[class_idx]).sum()
            )
            union = (
                valid_output_soft[class_idx].sum()
                + valid_target_one_hot[class_idx].sum()
            )
            dice_loss_class = 1 - (intersection + smooth) / (union + smooth)
            dice_loss_per_class.append(dice_loss_class)

        # 计算样本的平均Dice损失
        dice_loss_sample = torch.stack(dice_loss_per_class).mean()
        dice_losses.append(dice_loss_sample)
    # 计算批次的平均Dice损失
    dice_loss_batch = torch.stack(dice_losses).mean()
    return dice_loss_batch


def ce_loss(output, target):
    """
    计算加权交叉熵损失，忽略标签为-1的点。
    参数:
    - output: 模型输出，大小为[B, 3, H, W]。
    - target: 真实标签，大小为[B, H, W]，包含0, 1, 2和-1，-1需要被忽略。

    返回:
    - loss: 加权交叉熵损失的值。
    """
    weights = torch.tensor([1.0, 10.0, 10.0], device=output.device)
    loss = F.cross_entropy(output, target, weight=weights, ignore_index=-1)
    return loss
