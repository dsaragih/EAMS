import torch
import torch.nn as nn
import torch.nn.functional as F

from data.types import MeshData
from loguru import logger

from models.layers.egnn_conv import EGNN, EGNNmc
from models.layers.emnn_conv import MeshEGNN, MeshEGNNmc
from models.layers.emnn_pool import Pooling
from models.layers.emnn_unpool import Unpooling
from models.networks.encoders.norms import build_norm_layer, canonicalize_norm_type

from ..contracts import EncoderOutput


def log_stats(tensor: torch.Tensor, name: str):
    """Log basic statistics of a tensor for debugging."""
    logger.debug(
        f"{name} stats - shape: {tensor.shape}, min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}, std: {tensor.std().item():.4f}, has_nan: {torch.isnan(tensor).any()}"
    )


class EMNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edges_input_dim: int,
        width: int,
        width_mult: list[int],
        num_vector: int = 1,
        num_pooling_layers: int = None,
        hier: bool = False,
        scale_updown: int = 1,
        virtual_global_node: bool = False,
        detect_nan: bool = True,
        thickness_scale: float = 3.0,
        thickness_init: float = 1.0,
        pre_norm: bool = True,
        use_auxiliary_frame: bool = True,
        use_radial_edge_embedding: bool = False,
        conv_attention: bool = False,
        conv_type: str = "mesh_egnn",
        normalize_coord_diff: bool = False,
        coord_update_tanh: bool = False,
        norm_type: str = "stateless_batchnorm",
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.base_edges_input_dim = edges_input_dim
        self.use_radial_edge_embedding = use_radial_edge_embedding
        self.radial_edge_dim = 16 if use_radial_edge_embedding else 0
        self.edges_input_dim = self.base_edges_input_dim + self.radial_edge_dim
        self.width = width
        self.width_mult = width_mult
        self.num_vector = num_vector
        self.hier = hier
        self.scale_updown = scale_updown
        self.detect_nan = detect_nan
        self.virtual_global_node = virtual_global_node
        self.pre_norm = pre_norm
        self.use_auxiliary_frame = use_auxiliary_frame
        self.conv_attention = conv_attention
        self.conv_type = self._canonicalize_conv_type(conv_type)
        self.normalize_coord_diff = normalize_coord_diff
        self.coord_update_tanh = coord_update_tanh
        self.norm_type = canonicalize_norm_type(norm_type)

        if self.num_vector < 1:
            raise ValueError(f"num_vector must be >= 1, got: {self.num_vector}")
        if self.num_vector > 3:
            raise ValueError(
                "EMNN currently supports at most 3 coordinate channels: pos, normal, and one auxiliary frame"
            )
        if not self.use_auxiliary_frame and self.num_vector > 2:
            raise ValueError(
                "num_vector > 2 requires use_auxiliary_frame=True because the auxiliary frame is no longer routed through EMNN coordinates."
            )

        self.num_layers = len(width_mult) - 1
        if num_pooling_layers is not None:
            logger.info(
                f"Using {num_pooling_layers} pooling layers (hierarchical EMNN)"
            )
            if not hier:
                raise ValueError(
                    "num_pooling_layers specified but hier is False. Set hier=True for pooling layers."
                )
            if num_pooling_layers > self.num_layers:
                raise ValueError(
                    f"num_pooling_layers ({num_pooling_layers}) cannot be greater than number of conv layers ({self.num_layers})"
                )
            self.num_pooling_layers = num_pooling_layers
        else:
            self.num_pooling_layers = self.num_layers if hier else 0

        self.mc = num_vector > 1

        self.feat = nn.Linear(input_dim, width)
        self.input_norm = build_norm_layer(input_dim, self.norm_type)
        self.embed_norm = build_norm_layer(width, self.norm_type)
        self.thickness_init_mlp = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.thickness_tau = nn.Parameter(torch.tensor(thickness_init))
        self.thickness_scale = thickness_scale

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = width * width_mult[i]
            out_dim = width * width_mult[i + 1]
            num_vectors_in = num_vector if self.mc else 1
            num_vectors_out = num_vector if (self.mc and i < self.num_layers - 1) else 1
            layer_cls, layer_kwargs = self._build_conv_layer_spec(
                in_dim=in_dim,
                out_dim=out_dim,
                num_vectors_in=num_vectors_in,
                num_vectors_out=num_vectors_out,
                is_last_layer=(i == self.num_layers - 1),
            )
            self.conv_layers.append(layer_cls(in_dim, out_dim, in_dim, **layer_kwargs))
        self.output_dim = out_dim

        if self.virtual_global_node:
            logger.info("Adding virtual global node to EMNN")
            self.global_node_mlps = nn.ModuleList()
            for i in range(self.num_layers):
                dim_i = width * width_mult[i + 1]
                self.global_node_mlps.append(
                    nn.Sequential(
                        nn.Linear(dim_i, dim_i),
                        nn.SiLU(),
                        nn.Linear(dim_i, dim_i),
                    )
                )

        if self.hier:
            self.pooling_layers = nn.ModuleList()
            self.hier_dims = [
                int(self.output_dim * (self.scale_updown**i))
                for i in range(self.num_pooling_layers + 1)
            ]
            for i in range(self.num_pooling_layers):
                self.pooling_layers.append(
                    Pooling(
                        self.hier_dims[i],
                        self.hier_dims[i + 1],
                        ratio=0.25,
                        r=0.2,
                        global_pool=False,
                    )
                )
            self.unpooling_layers = nn.ModuleList()
            for i in range(self.num_pooling_layers - 1, -1, -1):
                in_dim = self.hier_dims[i + 1] + self.hier_dims[i]
                out_dim = self.hier_dims[i]
                self.unpooling_layers.append(Unpooling(in_dim, out_dim, k=3))

    def _canonicalize_conv_type(self, conv_type: str) -> str:
        normalized = conv_type.strip().lower().replace("-", "_")
        aliases = {
            "mesh": "mesh_egnn",
            "mesh_egnn": "mesh_egnn",
            "emnn": "mesh_egnn",
            "egnn": "egnn",
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported conv_type '{conv_type}'. Expected one of: mesh_egnn, egnn"
            )
        return aliases[normalized]

    def _build_conv_layer_spec(
        self,
        *,
        in_dim: int,
        out_dim: int,
        num_vectors_in: int,
        num_vectors_out: int,
        is_last_layer: bool,
    ) -> tuple[type[nn.Module], dict[str, object]]:
        common_kwargs: dict[str, object] = {
            "edges_in_d": self.edges_input_dim,
            "attention": self.conv_attention,
            "normalize": self.normalize_coord_diff,
            "coords_agg": "sum",
            "tanh": self.coord_update_tanh,
        }

        if self.conv_type == "mesh_egnn":
            layer_cls = MeshEGNNmc if self.mc else MeshEGNN
        else:
            layer_cls = EGNNmc if self.mc else EGNN

        if self.mc:
            common_kwargs["num_vectors_in"] = num_vectors_in
            common_kwargs["num_vectors_out"] = num_vectors_out
            common_kwargs["last_layer"] = is_last_layer

        return layer_cls, common_kwargs

    def _apply_virtual_global_node(
        self, x: torch.Tensor, batch: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        if not self.virtual_global_node:
            return x

        if x.ndim != 2:
            raise ValueError(
                f"virtual_global_node expects x with shape (N, C), got: {tuple(x.shape)}"
            )
        if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
            raise ValueError(
                f"virtual_global_node expects batch with shape (N,), got: {tuple(batch.shape)}"
            )
        if not (0 <= layer_idx < len(self.global_node_mlps)):
            raise ValueError(
                f"layer_idx out of range for global_node_mlps: {layer_idx}"
            )

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if num_graphs == 0:
            return x

        feat_dim = x.shape[-1]
        sum_x = x.new_zeros((num_graphs, feat_dim))
        sum_x.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=num_graphs).to(
            device=x.device, dtype=x.dtype
        )
        mean_x = sum_x / counts.clamp(min=1).unsqueeze(-1)

        global_update = self.global_node_mlps[layer_idx](mean_x)
        return x + global_update[batch]

    def _diagnose_nan_cause(self, x_stats, layer_idx):
        suggestions = []

        if x_stats["max"] > 1e4:
            suggestions.append(
                f"⚠️  Very large values before layer {layer_idx} (max={x_stats['max']:.2e}). Possible exploding gradients."
            )
            suggestions.append(
                "   → Try gradient clipping: trainer.gradient_clip_val=1.0"
            )
            suggestions.append("   → Or reduce learning rate")

        if x_stats["std"] > 1e3:
            suggestions.append(
                f"⚠️  High variance (std={x_stats['std']:.2e}). Consider switching model.encoder_config.norm_type=layernorm."
            )

        if abs(x_stats["mean"]) > 1e3:
            suggestions.append(
                f"⚠️  Large mean (mean={x_stats['mean']:.2e}). Check for numerical instability."
            )

        suggestions.append("\n💡 Common fixes:")
        suggestions.append("   1. Use gradient clipping: trainer.gradient_clip_val=1.0")
        suggestions.append("   2. Reduce learning rate (current lr * 0.1)")
        suggestions.append("   3. Check for division by zero in convolution layers")
        suggestions.append("   4. Verify input data normalization")
        suggestions.append("   5. Add batch/layer normalization")

        return "\n".join(suggestions)

    @property
    def required_batch_attributes(self) -> list[str]:
        req = ["edge_index", "pos", "batch", "x", "face"]
        if self.base_edges_input_dim > 0:
            req.append("edge_attr")
        if self.num_vector >= 2:
            req.append("normal")
        return req

    def _prepare_coordinate_inputs(
        self,
        data: MeshData,
        pos_origin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        auxiliary_frame, _ = self._get_auxiliary_frame(data)
        normal = getattr(data, "normal", None)

        if normal is not None:
            normal = normal.float()
            self._check_finite(
                normal,
                name="input normals",
                error_message="Non-finite values in input normals",
            )

        if not self.use_auxiliary_frame:
            auxiliary_frame = None

        if self.num_vector == 1:
            if auxiliary_frame is not None:
                raise ValueError(
                    "Input data contains an auxiliary frame but num_vector=1. Use the EMNN num_vector resolver or remove the cartesian frame feature from the feature set."
                )
            return pos_origin, normal, auxiliary_frame

        if normal is None:
            raise ValueError(
                f"Input data is missing required attribute 'normal' for num_vector={self.num_vector}"
            )

        if self.num_vector == 2:
            if auxiliary_frame is not None:
                raise ValueError(
                    "Input data contains an auxiliary frame but num_vector=2. Expected num_vector=3 when a cartesian frame feature is enabled."
                )
            return torch.stack([pos_origin, normal], dim=-1), normal, auxiliary_frame

        if auxiliary_frame is None:
            raise ValueError(
                "Input data is missing required auxiliary frame attribute for num_vector=3"
            )

        return (
            torch.stack([pos_origin, normal, auxiliary_frame], dim=-1),
            normal,
            auxiliary_frame,
        )

    def _get_auxiliary_frame(
        self, data: MeshData
    ) -> tuple[torch.Tensor | None, str | None]:
        canonical_frame = getattr(data, "canonical_frame", None)
        dental_frame = getattr(data, "dental_frame", None)
        auxiliary_frame = (
            canonical_frame if canonical_frame is not None else dental_frame
        )
        auxiliary_frame_name = (
            "canonical_frame"
            if canonical_frame is not None
            else "dental_frame" if dental_frame is not None else None
        )

        if auxiliary_frame is None:
            return None, auxiliary_frame_name

        auxiliary_frame = auxiliary_frame.float()
        self._check_finite(
            auxiliary_frame,
            name=f"input {auxiliary_frame_name}",
            error_message=f"Non-finite values in input {auxiliary_frame_name}",
        )
        return auxiliary_frame, auxiliary_frame_name

    def _normalize_features(
        self, tensor: torch.Tensor, norm: nn.Module
    ) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError(
                f"Feature normalization expects a rank-2 tensor, got shape {tuple(tensor.shape)}"
            )
        return norm(tensor)

    def _check_finite(
        self, tensor: torch.Tensor | None, name: str, error_message: str
    ) -> None:
        if tensor is None or not self.detect_nan:
            return
        finite_mask = torch.isfinite(tensor)
        if finite_mask.all():
            return

        logger.error(f"Non-finite values detected in {name}")
        logger.error(f"  shape: {tuple(tensor.shape)}")
        logger.error(f"  finite count: {finite_mask.sum().item()}/{tensor.numel()}")

        finite_values = tensor[finite_mask]
        if finite_values.numel() > 0:
            logger.error(
                "  finite stats - min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(
                    finite_values.min().item(),
                    finite_values.max().item(),
                    finite_values.mean().item(),
                    finite_values.std().item() if finite_values.numel() > 1 else 0.0,
                )
            )

        raise ValueError(error_message)

    def _embed_node_features(self, node_features: torch.Tensor) -> torch.Tensor:
        node_features = node_features.float()
        self._check_finite(
            node_features,
            name="input node features",
            error_message="Non-finite values in input node features",
        )

        if self.pre_norm:
            node_features = self._normalize_features(node_features, self.input_norm)

        x = self.feat(node_features)

        if self.detect_nan and not torch.isfinite(x).all():
            weight_is_finite = torch.isfinite(self.feat.weight).all().item()
            bias_is_finite = True
            if self.feat.bias is not None:
                bias_is_finite = torch.isfinite(self.feat.bias).all().item()

            logger.error(
                "Non-finite values detected immediately after the initial feature projection"
            )
            logger.error(
                f"  projection weight finite: {weight_is_finite}, bias finite: {bias_is_finite}"
            )
            if not weight_is_finite or not bias_is_finite:
                raise ValueError(
                    "Initial feature projection parameters became non-finite during training"
                )

        if not self.pre_norm:
            x = self._normalize_features(x, self.embed_norm)

        x = F.silu(x)
        self._check_finite(
            x,
            name="initial node embedding",
            error_message="NaN in initial features",
        )
        return x

    def _prepare_node_features(self, data: MeshData) -> torch.Tensor:
        node_features = data.x
        auxiliary_frame, _ = self._get_auxiliary_frame(data)

        if not self.use_auxiliary_frame and auxiliary_frame is not None:
            node_features = torch.cat([node_features.float(), auxiliary_frame], dim=-1)

        return self._embed_node_features(node_features)

    def _compute_edge_lengths(
        self, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0].long(), edge_index[1].long()
        return torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)

    def _rbf_expand(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.ndim != 2 or distances.shape[-1] != 1:
            raise ValueError(
                f"RBF expansion expects distances of shape (E, 1), got: {tuple(distances.shape)}"
            )
        if distances.shape[0] == 0:
            return distances.new_zeros((0, self.radial_edge_dim))

        max_distance = distances.detach().amax().clamp(min=1e-6)
        scaled = distances / max_distance
        centers = torch.linspace(
            0.0,
            1.0,
            self.radial_edge_dim,
            device=distances.device,
            dtype=distances.dtype,
        ).view(1, -1)

        if self.radial_edge_dim == 1:
            sigma = 1.0
        else:
            sigma = 1.0 / (self.radial_edge_dim - 1)

        return torch.exp(-0.5 * ((scaled - centers) / sigma) ** 2)

    def _prepare_edge_attr(self, data: MeshData) -> torch.Tensor | None:
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.float()
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            self._check_finite(
                edge_attr,
                name="input edge features",
                error_message="Non-finite values in input edge features",
            )

        if not self.use_radial_edge_embedding:
            return edge_attr

        edge_lengths = self._compute_edge_lengths(data.pos.float(), data.edge_index)
        radial_edge_attr = self._rbf_expand(edge_lengths)
        self._check_finite(
            radial_edge_attr,
            name="radial edge embedding",
            error_message="Non-finite values in radial edge embedding",
        )

        if edge_attr is None:
            edge_attr = radial_edge_attr
        else:
            edge_attr = torch.cat([edge_attr, radial_edge_attr], dim=-1)

        self._check_finite(
            edge_attr,
            name="prepared edge features",
            error_message="Non-finite values in prepared edge features",
        )
        return edge_attr

    def _prepare_forward_inputs(self, data: MeshData) -> dict[str, torch.Tensor | None]:
        for attr in self.required_batch_attributes:
            if not hasattr(data, attr):
                raise ValueError(f"Input data is missing required attribute '{attr}'")

        batch = data.batch
        pos_origin = data.pos.float()
        face = data.face

        self._check_finite(
            pos_origin,
            name="input positions",
            error_message="Non-finite values in input positions",
        )

        pos, normal, auxiliary_frame = self._prepare_coordinate_inputs(data, pos_origin)

        x = self._prepare_node_features(data)
        edge_attr = self._prepare_edge_attr(data)
        thickness, thickness_paired_idx, thickness_embed = self._init_thickness(
            data, x, normal
        )

        return {
            "batch": batch,
            "pos": pos,
            "pos_origin": pos_origin.clone(),
            "face": face,
            "normal": normal,
            "dental_frame": auxiliary_frame,
            "x": x,
            "edge_attr": edge_attr,
            "thickness": thickness,
            "thickness_paired_idx": thickness_paired_idx,
            "thickness_embed": thickness_embed,
        }

    def _compute_thickness_gate(self, thickness: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-self.thickness_scale * (thickness - self.thickness_tau))

    def _init_thickness(
        self, data: MeshData, x: torch.Tensor, normal: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        use_thickness = hasattr(data, "thickness") or hasattr(
            data, "thickness_paired_idx"
        )
        if not use_thickness:
            return None, None, None
        if not hasattr(data, "thickness") or not hasattr(data, "thickness_paired_idx"):
            raise ValueError(
                "Thickness support requires both `data.thickness` and `data.thickness_paired_idx`."
            )
        if normal is None:
            raise ValueError(
                "Thickness support requires `data.normal` to compute dot(n_i, n_T(i))."
            )

        thickness = data.thickness.to(device=x.device, dtype=x.dtype).view(-1)
        thickness_paired_idx = (
            data.thickness_paired_idx.to(device=x.device).long().view(-1)
        )

        num_nodes = x.shape[0]
        if thickness.numel() != num_nodes or thickness_paired_idx.numel() != num_nodes:
            raise ValueError(
                f"thickness and thickness_paired_idx must have shape (num_nodes,), got: {tuple(thickness.shape)} and {tuple(thickness_paired_idx.shape)}"
            )

        self_idx = torch.arange(num_nodes, device=x.device)
        invalid = (thickness_paired_idx < 0) | (thickness_paired_idx >= num_nodes)
        thickness_paired_idx = torch.where(invalid, self_idx, thickness_paired_idx)
        batch_idx = data.batch
        cross_graph = batch_idx[thickness_paired_idx] != batch_idx
        thickness_paired_idx = torch.where(cross_graph, self_idx, thickness_paired_idx)

        normal_unit = torch.nn.functional.normalize(
            normal.to(device=x.device, dtype=x.dtype), p=2, dim=-1, eps=1e-12
        )
        dot = (
            (normal_unit * normal_unit[thickness_paired_idx])
            .sum(dim=-1)
            .clamp(-1.0, 1.0)
        )

        init_feat = torch.stack([thickness, dot], dim=-1)
        thickness_embed = self.thickness_init_mlp(init_feat)
        return thickness, thickness_paired_idx, thickness_embed

    def _run_conv_layer(
        self,
        *,
        conv_layer: nn.Module,
        layer_idx: int,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        face: torch.Tensor,
        edge_attr: torch.Tensor | None,
        data: MeshData,
        thickness: torch.Tensor | None,
        thickness_embed: torch.Tensor | None,
        thickness_paired_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x_stats = {
            "min": x.min().item(),
            "max": x.max().item(),
            "mean": x.mean().item(),
            "std": x.std().item(),
            "has_nan": torch.isnan(x).any().item(),
        }

        if thickness_embed is None:
            x, pos = conv_layer(
                x,
                edge_index=data.edge_index.long(),
                coord=pos,
                edge_attr=edge_attr,
                face=face.long(),
            )
        else:
            thickness_scale = self._compute_thickness_gate(thickness)
            x, pos, thickness_embed = conv_layer(
                x,
                edge_index=data.edge_index.long(),
                coord=pos,
                edge_attr=edge_attr,
                face=face.long(),
                thickness_embed=thickness_embed,
                thickness_paired_idx=thickness_paired_idx,
                thickness_scale=thickness_scale.unsqueeze(-1),
            )

        if self.virtual_global_node:
            x = self._apply_virtual_global_node(x, batch=batch, layer_idx=layer_idx)

        if self.detect_nan and not torch.isfinite(x).all():
            logger.error(
                f"Non-finite values detected in model output after layer {layer_idx}!"
            )
            logger.error(f"  Output shape: {x.shape}")
            logger.error(
                f"  Finite count: {torch.isfinite(x).sum().item()}/{x.numel()}"
            )
            logger.error(
                f"  Input positions finite: {torch.isfinite(pos).all().item()}"
            )
            logger.error(f"x stats before layer {layer_idx}: {x_stats}")
            logger.error(self._diagnose_nan_cause(x_stats, layer_idx=layer_idx))
            raise ValueError("NaN detected in model output")

        return x, pos, thickness_embed

    def _run_hierarchy(
        self,
        *,
        x: torch.Tensor,
        pos_origin: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if not self.hier:
            return x

        pos_origin_curr, pos_curr, x_curr, batch_curr = (pos_origin, pos, x, batch)
        pool_results = [(pos_origin_curr, pos_curr, x_curr, batch_curr)]

        for pool_layer in self.pooling_layers:
            pos_origin_curr, pos_curr, x_curr, batch_curr = pool_layer(
                pos_origin_curr, pos_curr, x_curr, batch_curr
            )
            pool_results.append((pos_origin_curr, pos_curr, x_curr, batch_curr))

        x_curr = pool_results[-1][2]
        for i, unpool_layer in enumerate(self.unpooling_layers):
            pos_origin_from, _, _, batch_from = pool_results[-(i + 1)]
            pos_origin_to, _, x_to, batch_to = pool_results[-(i + 2)]
            x_curr = unpool_layer(
                pos_origin_from,
                x_curr,
                batch_from,
                pos_origin_to,
                x_to,
                batch_to,
            )

        if self.virtual_global_node:
            x_curr = self._apply_virtual_global_node(
                x_curr, batch=batch, layer_idx=len(self.global_node_mlps) - 1
            )

        return x_curr

    def forward(self, data: MeshData) -> EncoderOutput:
        inputs = self._prepare_forward_inputs(data)
        batch = inputs["batch"]
        pos = inputs["pos"]
        pos_origin = inputs["pos_origin"]
        face = inputs["face"]
        x = inputs["x"]
        edge_attr = inputs["edge_attr"]
        thickness = inputs["thickness"]
        thickness_paired_idx = inputs["thickness_paired_idx"]
        thickness_embed = inputs["thickness_embed"]

        for i, conv_layer in enumerate(self.conv_layers):
            x, pos, thickness_embed = self._run_conv_layer(
                conv_layer=conv_layer,
                layer_idx=i,
                x=x,
                pos=pos,
                batch=batch,
                face=face,
                edge_attr=edge_attr,
                data=data,
                thickness=thickness,
                thickness_embed=thickness_embed,
                thickness_paired_idx=thickness_paired_idx,
            )

        x = self._run_hierarchy(x=x, pos_origin=pos_origin, pos=pos, batch=batch)

        face_batch = None
        if face is not None:
            face_batch = (
                batch[face[0].long()]
                if face.numel() > 0
                else batch.new_empty((0,), dtype=batch.dtype)
            )

        return EncoderOutput(
            node_embedding=x,
            pos=pos,
            batch=batch,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            face=face,
            face_batch=face_batch,
        )
