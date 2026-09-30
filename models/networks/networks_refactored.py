"""
Refactored MeshEncoderDecoder Module

This module implements the nested resolution Mesh-Graph CNN architecture described in the paper.
The network processes liver meshes at multiple resolutions to segment anatomical landmarks.

Architecture Overview:
1. Mesh Pooling: Compress high-res mesh (22k edges) to low-res (3k edges)
2. Global Feature Extraction: DGCNN captures global structural features
3. Two-Level Propagation: "Dilation" and "Erosion" branches map predictions back to high-res
4. Anatomical Refinement: MeshConv-based network refines using topological details
5. Attention Fusion: Combines dilation and erosion branches for final prediction

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from models.layers.mesh_conv import MeshConv, MeshConv_high
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
from models.networks.networks import Attn

from .model_dgcnn import DGCNN_partseg
from loguru import logger


class MeshEncoderDecoderRefactored(nn.Module):
    """
    Nested Resolution Mesh-Graph CNN for anatomical landmark segmentation.

    This network combines global shape analysis (DGCNN) with local topological
    refinement (MeshConv) through a multi-stage architecture:

    Stage 1: Mesh Pooling & Global Feature Extraction
        - Pool high-resolution mesh (22,000 edges) → low-resolution (3,000 edges)
        - Extract global features using pretrained DGCNN

    Stage 2: Two-Level Propagation
        - Dilation branch: Liberal landmark proposal (sensitivity-focused)
        - Erosion branch: Conservative landmark proposal (precision-focused)

    Stage 3: Topological Refinement
        - Extract edge topology features (5 channels)
        - Incorporate mesh vertex count as global context
        - Hierarchical feature extraction with MeshConv

    Stage 4: Attention Fusion
        - Adaptive fusion of dilation and erosion features
        - Final segmentation prediction
    """

    # Architecture constants
    HIGH_RES_EDGES = 22000  # Original mesh resolution
    LOW_RES_EDGES = 3000  # Pooled mesh resolution
    NUM_CLASSES = 3  # Background, ligament, ridge
    EDGE_FEATURE_DIM = 5  # Dihedral angles, edge lengths, etc.
    POSITION_FEATURE_DIM = 17  # Spatial position features

    # Network hyperparameters
    EDGE_EMBED_DIM = 16  # Embedded edge feature dimension
    CONV1_OUT_DIM = 32  # First convolution output
    CONV2_OUT_DIM = 64  # Second convolution output
    CONV3_OUT_DIM = 64  # Third convolution output
    FUSION_DIM = 160  # Concatenated feature dimension (32+64+64)
    REFINE1_DIM = 64  # First refinement layer
    REFINE2_DIM = 32  # Second refinement layer

    # DGCNN parameters
    DGCNN_INPUT_DIM = 40  # DGCNN expects 40-dim input
    DGCNN_PART_CATEGORIES = (
        16  # Number of part categories (legacy from pretrained model)
    )

    # Normalization constants for vertex count feature
    VERTEX_COUNT_MEAN = 3000.0
    VERTEX_COUNT_STD = 19000.0

    # Dilation and erosion thresholds
    DILATION_THRESHOLD = 0.0  # Probability threshold for dilation (liberal)
    EROSION_THRESHOLD = 0.9  # Probability threshold for erosion (conservative)

    def __init__(
        self,
        pools,
        down_convs,
        up_convs,
        blocks=0,
        transfer_data=True,
        dgcnn_checkpoint_path=None,
    ):
        """
        Initialize the nested resolution encoder-decoder network.

        Args:
            pools: Pooling layer configuration (legacy, kept for compatibility)
            down_convs: Downsampling convolution config (legacy)
            up_convs: Upsampling convolution config (legacy)
            blocks: Number of residual blocks (legacy)
            transfer_data: Whether to transfer data between stages (legacy)
            dgcnn_checkpoint_path: Path to pretrained DGCNN model weights. If None,
                                   the DGCNN branch is randomly initialized and trained
                                   jointly with the rest of the model.
        """
        super(MeshEncoderDecoderRefactored, self).__init__()
        self.transfer_data = transfer_data
        self.dgcnn_checkpoint_path = dgcnn_checkpoint_path
        self.dgcnn_is_pretrained = dgcnn_checkpoint_path is not None

        # Stage 1: Mesh Pooling and Unpooling
        self.pool = MeshPool(self.LOW_RES_EDGES)
        self.unpool = MeshUnpool(self.HIGH_RES_EDGES)

        # Vertex count encoder (global context feature)
        self.count_encoder = nn.Sequential(
            nn.Linear(1, 8, bias=True),
            nn.Linear(8, self.NUM_CLASSES, bias=True),
            nn.ELU(),
        )

        # Edge feature embedding
        self.edge_feature_embed = nn.Conv1d(
            self.EDGE_FEATURE_DIM, self.EDGE_EMBED_DIM, kernel_size=1
        )

        # Initialize DGCNN either from a checkpoint or from scratch.
        self.dgcnn_model = self._initialize_dgcnn_model()

        # Stage 3: Dilation Branch (Liberal Proposals)
        # Processes dilation landmarks with hierarchical MeshConv
        self.dilation_conv1 = MeshConv_high(
            self.EDGE_EMBED_DIM
            + self.NUM_CLASSES
            + self.NUM_CLASSES,  # 16 + 3 + 3 = 22
            self.CONV1_OUT_DIM,
        )
        self.dilation_conv2 = MeshConv(self.CONV1_OUT_DIM, self.CONV2_OUT_DIM)
        self.dilation_conv3 = MeshConv(self.CONV2_OUT_DIM, self.CONV3_OUT_DIM)

        self.dilation_bn1 = nn.InstanceNorm1d(self.CONV1_OUT_DIM)
        self.dilation_bn2 = nn.InstanceNorm1d(self.CONV2_OUT_DIM)
        self.dilation_bn3 = nn.InstanceNorm1d(self.CONV3_OUT_DIM)

        # Stage 3: Erosion Branch (Conservative Proposals)
        # Processes erosion landmarks with same architecture
        self.erosion_conv1 = MeshConv_high(
            self.EDGE_EMBED_DIM + self.NUM_CLASSES + self.NUM_CLASSES,
            self.CONV1_OUT_DIM,
        )
        self.erosion_conv2 = MeshConv(self.CONV1_OUT_DIM, self.CONV2_OUT_DIM)
        self.erosion_conv3 = MeshConv(self.CONV2_OUT_DIM, self.CONV3_OUT_DIM)

        self.erosion_bn1 = nn.InstanceNorm1d(self.CONV1_OUT_DIM)
        self.erosion_bn2 = nn.InstanceNorm1d(self.CONV2_OUT_DIM)
        self.erosion_bn3 = nn.InstanceNorm1d(self.CONV3_OUT_DIM)

        # Auxiliary prediction heads (for deep supervision)
        self.dilation_head = nn.Conv1d(self.FUSION_DIM, self.NUM_CLASSES, kernel_size=1)
        self.erosion_head = nn.Conv1d(self.FUSION_DIM, self.NUM_CLASSES, kernel_size=1)

        # Stage 4: Attention Fusion Module
        # Import from networks.py attention modules (assuming Attn is defined elsewhere)
        self.attention_fusion = Attn(self.FUSION_DIM)

        # Final refinement layers
        self.refine1 = nn.Sequential(
            nn.Conv1d(self.FUSION_DIM, self.REFINE1_DIM, kernel_size=1, bias=False),
            nn.InstanceNorm1d(self.REFINE1_DIM),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.refine2 = nn.Sequential(
            nn.Conv1d(self.REFINE1_DIM, self.REFINE2_DIM, kernel_size=1, bias=False),
            nn.InstanceNorm1d(self.REFINE2_DIM),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Final prediction head
        self.final_head = nn.Conv1d(self.REFINE2_DIM, self.NUM_CLASSES, kernel_size=1)

    def _build_dgcnn_model(self):
        return DGCNN_partseg(self.DGCNN_INPUT_DIM, self.NUM_CLASSES)

    def _load_dgcnn_checkpoint(
        self, dgcnn_model: nn.Module, checkpoint_path: str
    ) -> nn.Module:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        try:
            dgcnn_model.load_state_dict(state_dict)
            return dgcnn_model
        except RuntimeError as exc:
            if not isinstance(state_dict, dict):
                raise

        if any(key.startswith("module.") for key in state_dict.keys()):
            stripped_state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }
            dgcnn_model.load_state_dict(stripped_state_dict)
            return dgcnn_model

        raise exc

    def _initialize_dgcnn_model(self):
        """
        Initialize the DGCNN branch.

        Returns:
            DGCNN model loaded from checkpoint and frozen when a checkpoint path
            is provided, otherwise a randomly initialized trainable DGCNN.
        """
        dgcnn_model = self._build_dgcnn_model()
        if self.dgcnn_checkpoint_path is None:
            logger.warning(
                "MeshEncoderDecoderRefactored initialized without a DGCNN checkpoint. "
                "The DGCNN branch will be trained from scratch."
            )
            return dgcnn_model

        dgcnn_model = self._load_dgcnn_checkpoint(
            dgcnn_model,
            checkpoint_path=self.dgcnn_checkpoint_path,
        )
        dgcnn_model.requires_grad_(False)
        dgcnn_model.eval()
        return dgcnn_model

    def _encode_labels_to_onehot(self, labels):
        """
        Convert labels to one-hot encoding, handling padding (-1 values).

        Labels: -1 (padding), 0 (background), 1 (ligament), 2 (ridge)
        Mapping: -1→0, 0→1, 1→2, 2→3 for one-hot encoding

        Args:
            labels: Tensor of shape (B, N) with values in {-1, 0, 1, 2}

        Returns:
            One-hot tensor of shape (B, 3, N) excluding padding class
        """
        if labels.max() <= 1:
            # No actual labels, return zeros
            return torch.zeros(
                labels.shape[0],
                self.NUM_CLASSES,
                labels.shape[1],
                device=labels.device,
                dtype=torch.int,
            )

        # Map labels: -1→0, 0→1, 1→2, 2→3
        mapped_indices = labels + 1

        # One-hot encode (4 classes including padding)
        one_hot = F.one_hot(mapped_indices, num_classes=4)
        one_hot = one_hot.transpose(1, 2)  # (B, 4, N)

        # Remove padding class, keep only real classes
        return one_hot[:, 1:, :]  # (B, 3, N)

    def _compute_vertex_count_features(self, labels):
        """
        Compute normalized vertex count feature for each sample.

        This provides global context about mesh size/complexity.

        Args:
            labels: Tensor of shape (B, N) with padding marked as -1

        Returns:
            Tensor of shape (B, 3, 1) with normalized count features
        """
        batch_size = labels.shape[0]
        label_counts = []

        for batch_i in range(batch_size):
            # Count non-padding vertices
            count = torch.count_nonzero(labels[batch_i] != -1)
            label_counts.append(count.item())

        label_counts = (
            torch.tensor(label_counts, device=labels.device).unsqueeze(1).float()
        )

        # Normalize using dataset statistics
        normalized_counts = (
            label_counts - self.VERTEX_COUNT_MEAN
        ) / self.VERTEX_COUNT_STD

        # Encode through MLP
        count_features = self.count_encoder(normalized_counts)  # (B, 3)

        return count_features

    def _create_landmark_proposals(self, probabilities, threshold):
        """
        Create binary landmark proposals from DGCNN probabilities.

        Args:
            probabilities: Soft predictions from DGCNN (B, 3, N)
            threshold: Probability threshold for binarization

        Returns:
            Binary tensor (B, 3, N) with 0/1 values
        """
        proposals = probabilities.clone()
        proposals[proposals > threshold] = 1
        proposals[proposals != 1] = 0
        return proposals.round()

    def _extract_hierarchical_features(
        self, input_features, meshes, conv1, conv2, conv3, bn1, bn2, bn3
    ):
        """
        Extract hierarchical features using MeshConv layers.

        Args:
            input_features: Input tensor (B, C, N)
            meshes: List of mesh objects
            conv1, conv2, conv3: Convolution layers
            bn1, bn2, bn3: Batch normalization layers

        Returns:
            Concatenated multi-scale features (B, 32+64+64, N)
        """
        # Level 1: 32 channels
        fea = conv1(input_features, meshes).squeeze(-1)
        fea = bn1(fea)
        fea_0 = F.leaky_relu(fea, negative_slope=0.2)

        # Level 2: 64 channels
        fea = conv2(fea_0, meshes).squeeze(-1)
        fea = bn2(fea)
        fea_1 = F.leaky_relu(fea, negative_slope=0.2)

        # Level 3: 64 channels
        fea = conv3(fea_1, meshes).squeeze(-1)
        fea = bn3(fea)
        fea_2 = F.leaky_relu(fea, negative_slope=0.2)

        # Concatenate multi-scale features
        return torch.cat((fea_0, fea_1, fea_2), dim=1)  # (B, 160, N)

    def forward(self, x, labels, meshes):
        """
        Forward pass through the nested resolution network.

        Args:
            x: Input edge features (B, 5+17, N) where:
                - First 5 channels: Edge features (angles, lengths, etc.)
                - Next 17 channels: Position/coordinate features
            labels: Ground truth labels (B, N) with values {-1, 0, 1, 2}
            meshes: List of Mesh objects for convolution operations

        Returns:
            Tuple containing:
            - final_pred: Final segmentation prediction (B, 3, N)
            - dgcnn_pred: Raw DGCNN output (B, 3, ~3000)
            - dgcnn_pred_copy: Copy for loss computation
            - propagated_probs: Propagated probabilities to high-res (B, 3, N)
            - dilation_proposals: Dilation landmark proposals (B, 3, N)
            - erosion_proposals: Erosion landmark proposals (B, 3, N)
            - dilation_pred: Dilation branch prediction (B, 3, N)
            - erosion_pred: Erosion branch prediction (B, 3, N)
            - timing_info: Timing information for profiling
            - attention_weights_a: Dilation branch attention weights
            - attention_weights_b: Erosion branch attention weights
        """
        device = x.device

        # ============= Stage 1: Mesh Pooling & Feature Preparation =============

        # Split input features
        edge_features = x[:, : self.EDGE_FEATURE_DIM, :]  # (B, 5, N)
        position_features = x[:, self.EDGE_FEATURE_DIM :, :]  # (B, 3, N)

        # Encode labels to one-hot for pooling
        label_onehot = self._encode_labels_to_onehot(labels)  # (B, 3, N)

        # Concatenate position features with labels for pooling
        pooling_input = torch.cat((position_features, label_onehot), dim=1)  # (B, 6, N)

        # Pool to low resolution
        pooled_features = self.pool(
            pooling_input, pooling_input, meshes
        )  # (B, 6, ~3000)

        # Separate pooled position and labels
        pooled_positions = pooled_features[:, : -self.NUM_CLASSES, :]
        pooled_labels = pooled_features[:, -self.NUM_CLASSES :, :]

        # ============= Stage 2: Global Feature Extraction with DGCNN =============

        start_time_dgcnn = time.time()

        if self.dgcnn_is_pretrained:
            self.dgcnn_model.eval()

        # Prepare dummy category labels for DGCNN (legacy requirement)
        batch_size = pooled_positions.shape[0]
        category_labels = np.zeros((batch_size, self.DGCNN_PART_CATEGORIES))
        category_labels[:, 0] = 1  # Assign to category 0
        category_labels = torch.from_numpy(category_labels.astype(np.float32)).to(
            device
        )

        # DGCNN inference
        dgcnn_predictions = self.dgcnn_model(
            pooled_positions, category_labels
        )  # (B, 3, ~3000)

        # Convert to probabilities
        dgcnn_probs = torch.softmax(dgcnn_predictions, dim=1)

        # Convert to discrete predictions
        predicted_classes = torch.argmax(dgcnn_probs, dim=1)
        dgcnn_onehot = torch.zeros_like(dgcnn_predictions).scatter_(
            1, predicted_classes.unsqueeze(1), 1
        )

        dgcnn_time = time.time() - start_time_dgcnn

        # ============= Stage 3: Two-Level Propagation =============

        # Combine soft and hard predictions for unpooling
        unpool_input = torch.cat((dgcnn_probs, dgcnn_onehot), dim=1)  # (B, 6, ~3000)

        # Unpool to high resolution
        unpooled_features = self.unpool(unpool_input, meshes)  # (B, 6, N)

        # Separate propagated probabilities and one-hot
        propagated_probs = unpooled_features[:, : self.NUM_CLASSES, :]
        propagated_onehot = unpooled_features[:, self.NUM_CLASSES :, :]

        # Create dilation and erosion proposals
        dilation_proposals = self._create_landmark_proposals(
            propagated_onehot, self.DILATION_THRESHOLD
        )
        erosion_proposals = self._create_landmark_proposals(
            propagated_onehot, self.EROSION_THRESHOLD
        )

        # ============= Stage 4: Topological Refinement =============

        start_time_refine = time.time()

        # Embed edge features
        embedded_edges = self.edge_feature_embed(edge_features)  # (B, 16, N)

        # Compute vertex count features
        count_features = self._compute_vertex_count_features(labels)  # (B, 3)
        count_features = count_features.unsqueeze(2).repeat(
            1, 1, self.HIGH_RES_EDGES
        )  # (B, 3, N)

        # Prepare inputs for dilation and erosion branches
        dilation_input = torch.cat(
            (embedded_edges, count_features, dilation_proposals), dim=1
        )  # (B, 22, N)

        erosion_input = torch.cat(
            (embedded_edges, count_features, erosion_proposals), dim=1
        )  # (B, 22, N)

        # Extract hierarchical features for both branches
        dilation_features = self._extract_hierarchical_features(
            dilation_input,
            meshes,
            self.dilation_conv1,
            self.dilation_conv2,
            self.dilation_conv3,
            self.dilation_bn1,
            self.dilation_bn2,
            self.dilation_bn3,
        )  # (B, 160, N)

        erosion_features = self._extract_hierarchical_features(
            erosion_input,
            meshes,
            self.erosion_conv1,
            self.erosion_conv2,
            self.erosion_conv3,
            self.erosion_bn1,
            self.erosion_bn2,
            self.erosion_bn3,
        )  # (B, 160, N)

        refine_time = time.time() - start_time_refine

        # Auxiliary predictions for deep supervision
        dilation_pred = self.dilation_head(dilation_features)  # (B, 3, N)
        erosion_pred = self.erosion_head(erosion_features)  # (B, 3, N)

        # ============= Stage 5: Attention Fusion =============

        # Adaptive fusion of dilation and erosion features
        fused_features, attention_weights_a, attention_weights_b = (
            self.attention_fusion(dilation_features, erosion_features, meshes)
        )

        # Final refinement
        refined_features = self.refine1(fused_features)  # (B, 64, N)
        refined_features = self.refine2(refined_features)  # (B, 32, N)

        # Final prediction
        final_pred = self.final_head(refined_features)  # (B, 3, N)

        # ============= Return All Outputs =============

        timing_info = dgcnn_time + refine_time

        return (
            final_pred,  # Final segmentation
            dgcnn_predictions,  # DGCNN raw output
            dgcnn_predictions,  # Copy for loss computation
            propagated_probs,  # Propagated soft predictions
            dilation_proposals,  # Dilation landmark proposals
            erosion_proposals,  # Erosion landmark proposals
            dilation_pred,  # Dilation branch prediction
            erosion_pred,  # Erosion branch prediction
            timing_info,  # Timing information
            attention_weights_a,  # Dilation attention weights
            attention_weights_b,  # Erosion attention weights
        )

    def __call__(self, x, labels, meshes):
        """Callable interface for forward pass."""
        return self.forward(x, labels, meshes)
