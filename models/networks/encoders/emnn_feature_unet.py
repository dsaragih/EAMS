import torch
import torch.nn as nn
from torch_geometric.nn import radius

from data.types import MeshData
from models.layers.emnn_feature_pool import FeaturePooling
from models.layers.emnn_feature_unpool import FeatureUnpooling
from models.networks.encoders.norms import build_norm_layer
from models.layers.emnn_conv import MeshEGNN, MeshEGNNmc
from models.utils import aggregate_max, aggregate_mean, aggregate_sum

from ..contracts import EncoderOutput
from .emnn import EMNN


class CoarseGraphBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        radius_value: float,
        max_num_neighbors: int = 64,
        aggr: str = "mean",
        dropout: float = 0.0,
        norm_type: str = "stateless_batchnorm",
    ):
        super().__init__()
        if aggr not in {"max", "mean", "sum"}:
            raise ValueError(
                f"aggr must be one of {{'max', 'mean', 'sum'}}, got: {aggr}"
            )

        self.radius_value = radius_value
        self.max_num_neighbors = max_num_neighbors
        self.aggr = aggr
        self.norm = build_norm_layer(dim, norm_type)
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def _aggregate(
        self, src: torch.Tensor, index: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        if self.aggr == "max":
            return aggregate_max(src, index, dim_size)
        if self.aggr == "mean":
            return aggregate_mean(src, index, dim_size)
        return aggregate_sum(src, index, dim_size)

    def forward(
        self,
        x: torch.Tensor,
        pos_origin: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x

        row, col = radius(
            pos_origin,
            pos_origin,
            self.radius_value,
            batch,
            batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        valid = row != col
        row = row[valid]
        col = col[valid]
        if row.numel() == 0:
            return x

        rel = pos_origin[col] - pos_origin[row]
        x_norm = self.norm(x)
        msg = self.edge_mlp(torch.cat([x_norm[row], x_norm[col], rel], dim=-1))
        agg = self._aggregate(msg, row, x.shape[0])
        update = self.node_mlp(torch.cat([x_norm, agg], dim=-1))
        return x + update


class EMNNFeatureUNet(EMNN):
    def __init__(
        self,
        input_dim: int,
        edges_input_dim: int,
        width: int,
        width_mult: list[int],
        num_vector: int = 1,
        tier_layer_count: int | None = None,
        decoder_tier_layer_count: int | None = None,
        pool_layers_per_tier: int | None = None,
        pool_ratios: list[float] | None = None,
        pool_radii: list[float] | None = None,
        unpool_ks: list[int] | None = None,
        coarse_radii: list[float] | None = None,
        use_feature_scores: bool = True,
        pool_max_num_neighbors: int = 64,
        pool_random_start: bool = True,
        coarse_max_num_neighbors: int = 64,
        coarse_block_dropout: float = 0.0,
        virtual_global_node: bool = False,
        detect_nan: bool = True,
        thickness_scale: float = 3.0,
        pooling_aggr: str = "max",
        pre_norm: bool = True,
        use_radial_edge_embedding: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            edges_input_dim=edges_input_dim,
            width=width,
            width_mult=width_mult,
            num_vector=num_vector,
            hier=False,
            scale_updown=1,
            virtual_global_node=virtual_global_node,
            detect_nan=detect_nan,
            thickness_scale=thickness_scale,
            pre_norm=pre_norm,
            use_radial_edge_embedding=use_radial_edge_embedding,
            **kwargs,
        )

        self._init_tiered_feature_unet(
            width=width,
            width_mult=width_mult,
            tier_layer_count=tier_layer_count,
            decoder_tier_layer_count=decoder_tier_layer_count,
            pool_layers_per_tier=pool_layers_per_tier,
            pool_ratios=pool_ratios,
            pool_radii=pool_radii,
            unpool_ks=unpool_ks,
            coarse_radii=coarse_radii,
            use_feature_scores=use_feature_scores,
            pool_max_num_neighbors=pool_max_num_neighbors,
            pool_random_start=pool_random_start,
            coarse_max_num_neighbors=coarse_max_num_neighbors,
            coarse_block_dropout=coarse_block_dropout,
            pooling_aggr=pooling_aggr,
        )

    def _expand_transition_values(
        self,
        values: list[float] | list[int] | None,
        counts: list[int],
        default_factory,
        name: str,
    ) -> list[float] | list[int]:
        total = sum(counts)
        if values is None:
            return [default_factory(i) for i in range(total)]
        if len(values) == total:
            return list(values)
        if len(values) == len(counts):
            expanded = []
            for value, count in zip(values, counts):
                expanded.extend([value] * count)
            return expanded
        raise ValueError(
            f"{name} must have length sum(pool_layers_per_tier) ({total}) or len(pool_layers_per_tier) ({len(counts)}), got: {len(values)}"
        )

    def _repeat_scalar(
        self, value: int | None, length: int, default_value: int
    ) -> list[int]:
        repeated_value = default_value if value is None else value
        if repeated_value < 1:
            raise ValueError("tier and pooling counts must be >= 1")
        return [repeated_value] * length

    def _build_global_node_mlp(self, dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _apply_tier_global_node(
        self, x: torch.Tensor, batch: torch.Tensor, tier_idx: int
    ) -> torch.Tensor:
        if not self.virtual_global_node:
            return x

        if x.ndim != 2:
            raise ValueError(
                f"tier global node expects x with shape (N, C), got: {tuple(x.shape)}"
            )
        if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
            raise ValueError(
                f"tier global node expects batch with shape (N,), got: {tuple(batch.shape)}"
            )
        if not (0 <= tier_idx < len(self.tier_global_node_mlps)):
            raise ValueError(
                f"tier_idx out of range for tier_global_node_mlps: {tier_idx}"
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
        global_update = self.tier_global_node_mlps[tier_idx](mean_x)
        return x + global_update[batch]

    def _make_fine_block(self, dim: int, last_layer: bool = False) -> nn.Module:
        layer_cls = MeshEGNNmc if self.mc else MeshEGNN
        num_vectors_out = 1 if (self.mc and last_layer) else self.num_vector
        return layer_cls(
            dim,
            dim,
            dim,
            edges_in_d=self.edges_input_dim,
            attention=self.conv_attention,
            normalize=self.normalize_coord_diff,
            coords_agg="sum",
            tanh=self.coord_update_tanh,
            num_vectors_in=self.num_vector if self.mc else 1,
            num_vectors_out=num_vectors_out if self.mc else 1,
            last_layer=last_layer,
        )

    def _init_tiered_feature_unet(
        self,
        *,
        width: int,
        width_mult: list[int],
        tier_layer_count: int | None,
        decoder_tier_layer_count: int | None,
        pool_layers_per_tier: int | None,
        pool_ratios: list[float] | None,
        pool_radii: list[float] | None,
        unpool_ks: list[int] | None,
        coarse_radii: list[float] | None,
        use_feature_scores: bool,
        pool_max_num_neighbors: int,
        pool_random_start: bool,
        coarse_max_num_neighbors: int,
        coarse_block_dropout: float,
        pooling_aggr: str,
    ) -> None:
        self.tier_dims = [width * mult for mult in width_mult]
        if not self.tier_dims or self.tier_dims[0] != width:
            raise ValueError(
                "tiered EMNNFeatureUNet expects width_mult[0] == 1 so the first tier matches the projected feature width"
            )
        # The tiered U-Net decodes back to the finest tier, so the decoder must
        # be instantiated against the first tier width rather than the base EMNN
        # final conv width inherited from `EMNN.__init__`.
        self.output_dim = self.tier_dims[0]

        self.num_tiers = len(self.tier_dims)
        if self.num_tiers < 2:
            raise ValueError("tiered EMNNFeatureUNet requires at least two tiers")

        encoder_tier_depth = 1 if tier_layer_count is None else tier_layer_count
        decoder_tier_depth = (
            encoder_tier_depth
            if decoder_tier_layer_count is None
            else decoder_tier_layer_count
        )
        pooling_layers = 1 if pool_layers_per_tier is None else pool_layers_per_tier

        tier_layer_counts = self._repeat_scalar(encoder_tier_depth, self.num_tiers, 1)
        decoder_tier_layer_counts = self._repeat_scalar(
            decoder_tier_depth,
            self.num_tiers - 1,
            encoder_tier_depth,
        )
        pool_layers_per_tier = self._repeat_scalar(
            pooling_layers,
            self.num_tiers - 1,
            1,
        )

        expanded_pool_ratios = self._expand_transition_values(
            pool_ratios,
            pool_layers_per_tier,
            lambda _: 0.5,
            "pool_ratios",
        )
        expanded_pool_radii = self._expand_transition_values(
            pool_radii,
            pool_layers_per_tier,
            lambda i: 0.15 * (i + 1),
            "pool_radii",
        )
        expanded_unpool_ks = self._expand_transition_values(
            unpool_ks,
            pool_layers_per_tier,
            lambda _: 3,
            "unpool_ks",
        )

        if coarse_radii is None:
            coarse_radii = []
            cursor = 0
            for count in pool_layers_per_tier:
                coarse_radii.append(expanded_pool_radii[cursor + count - 1])
                cursor += count
        if len(coarse_radii) != self.num_tiers - 1:
            raise ValueError(
                f"coarse_radii must have length {self.num_tiers - 1}, got: {len(coarse_radii)}"
            )

        self.tier_layer_counts = tier_layer_counts
        self.decoder_tier_layer_counts = decoder_tier_layer_counts
        self.pool_layers_per_tier = pool_layers_per_tier
        self.tier_global_node_mlps = nn.ModuleList(
            self._build_global_node_mlp(dim) for dim in self.tier_dims
        )

        self.encoder_tier_blocks = nn.ModuleList()
        for tier_idx, (tier_dim, num_blocks) in enumerate(
            zip(self.tier_dims, tier_layer_counts)
        ):
            blocks = nn.ModuleList()
            if tier_idx == 0:
                for _ in range(num_blocks):
                    blocks.append(self._make_fine_block(tier_dim, last_layer=False))
            else:
                for _ in range(num_blocks):
                    blocks.append(
                        CoarseGraphBlock(
                            dim=tier_dim,
                            radius_value=coarse_radii[tier_idx - 1],
                            max_num_neighbors=coarse_max_num_neighbors,
                            aggr=pooling_aggr,
                            dropout=coarse_block_dropout,
                            norm_type=self.norm_type,
                        )
                    )
            self.encoder_tier_blocks.append(blocks)

        self.decoder_tier_blocks = nn.ModuleList()
        for tier_idx in range(self.num_tiers - 1):
            tier_dim = self.tier_dims[tier_idx]
            num_blocks = decoder_tier_layer_counts[tier_idx]
            blocks = nn.ModuleList()
            if tier_idx == 0:
                for block_idx in range(num_blocks):
                    blocks.append(
                        self._make_fine_block(
                            tier_dim,
                            last_layer=block_idx == num_blocks - 1,
                        )
                    )
            else:
                for _ in range(num_blocks):
                    blocks.append(
                        CoarseGraphBlock(
                            dim=tier_dim,
                            radius_value=coarse_radii[tier_idx - 1],
                            max_num_neighbors=coarse_max_num_neighbors,
                            aggr=pooling_aggr,
                            dropout=coarse_block_dropout,
                            norm_type=self.norm_type,
                        )
                    )
            self.decoder_tier_blocks.append(blocks)

        self.tier_pool_layers = nn.ModuleList()
        self.tier_unpool_layers = nn.ModuleList()
        cursor = 0
        for tier_idx, repeat_count in enumerate(pool_layers_per_tier):
            transition_dims = [self.tier_dims[tier_idx]] + [
                self.tier_dims[tier_idx + 1]
            ] * repeat_count
            pool_modules = nn.ModuleList()
            unpool_modules = nn.ModuleList()
            for step_idx in range(repeat_count):
                pool_modules.append(
                    FeaturePooling(
                        in_dim=transition_dims[step_idx],
                        out_dim=transition_dims[step_idx + 1],
                        ratio=expanded_pool_ratios[cursor],
                        r=expanded_pool_radii[cursor],
                        max_num_neighbors=pool_max_num_neighbors,
                        aggr=pooling_aggr,
                        random_start=pool_random_start,
                        use_feature_scores=use_feature_scores,
                    )
                )
                unpool_modules.append(
                    FeatureUnpooling(
                        coarse_dim=transition_dims[step_idx + 1],
                        skip_dim=transition_dims[step_idx],
                        out_dim=transition_dims[step_idx],
                        k=expanded_unpool_ks[cursor],
                    )
                )
                cursor += 1
            self.tier_pool_layers.append(pool_modules)
            self.tier_unpool_layers.append(unpool_modules)

    def _run_fine_block(
        self,
        *,
        conv_layer: nn.Module,
        layer_idx: int,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        tier_idx: int | None,
        face: torch.Tensor,
        edge_attr: torch.Tensor | None,
        data: MeshData,
        thickness: torch.Tensor | None,
        thickness_embed: torch.Tensor | None,
        thickness_paired_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
            x = self._apply_tier_global_node(x, batch=batch, tier_idx=tier_idx)
        return x, pos, thickness_embed

    def _run_tiered_feature_unet(
        self,
        *,
        data: MeshData,
        pos_origin: torch.Tensor,
        pos: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor,
        face: torch.Tensor,
        edge_attr: torch.Tensor | None,
        thickness: torch.Tensor | None,
        thickness_embed: torch.Tensor | None,
        thickness_paired_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pos_origin_curr = pos_origin
        pos_curr = pos
        x_curr = x
        batch_curr = batch
        skip_stack: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        fine_layer_idx = 0

        for block in self.encoder_tier_blocks[0]:
            x_curr, pos_curr, thickness_embed = self._run_fine_block(
                conv_layer=block,
                layer_idx=fine_layer_idx,
                x=x_curr,
                pos=pos_curr,
                batch=batch_curr,
                tier_idx=0,
                face=face,
                edge_attr=edge_attr,
                data=data,
                thickness=thickness,
                thickness_embed=thickness_embed,
                thickness_paired_idx=thickness_paired_idx,
            )
            fine_layer_idx += 1

        for tier_idx in range(self.num_tiers - 1):
            for pool_layer in self.tier_pool_layers[tier_idx]:
                skip_stack.append((pos_origin_curr, pos_curr, x_curr, batch_curr))
                pooled = pool_layer(pos_origin_curr, pos_curr, x_curr, batch_curr)
                pos_origin_curr = pooled["pos_origin"]
                pos_curr = pooled["pos"]
                x_curr = pooled["x"]
                batch_curr = pooled["batch"]

            for block in self.encoder_tier_blocks[tier_idx + 1]:
                x_curr = block(x_curr, pos_origin_curr, batch_curr)
                x_curr = self._apply_tier_global_node(
                    x_curr, batch=batch_curr, tier_idx=tier_idx + 1
                )

        self._check_finite(
            x_curr,
            name="tiered feature bottleneck branch",
            error_message="NaN detected in tiered feature bottleneck branch",
        )

        for tier_idx in range(self.num_tiers - 2, -1, -1):
            for unpool_layer in reversed(self.tier_unpool_layers[tier_idx]):
                pos_origin_skip, pos_skip, x_skip, batch_skip = skip_stack.pop()
                x_curr = unpool_layer(
                    pos_origin_curr,
                    x_curr,
                    batch_curr,
                    pos_origin_skip,
                    x_skip,
                    batch_skip,
                )
                pos_origin_curr = pos_origin_skip
                pos_curr = pos_skip
                batch_curr = batch_skip

            for block in self.decoder_tier_blocks[tier_idx]:
                if tier_idx == 0:
                    x_curr, pos_curr, thickness_embed = self._run_fine_block(
                        conv_layer=block,
                        layer_idx=fine_layer_idx,
                        x=x_curr,
                        pos=pos_curr,
                        batch=batch_curr,
                        tier_idx=0,
                        face=face,
                        edge_attr=edge_attr,
                        data=data,
                        thickness=thickness,
                        thickness_embed=thickness_embed,
                        thickness_paired_idx=thickness_paired_idx,
                    )
                    fine_layer_idx += 1
                else:
                    x_curr = block(x_curr, pos_origin_curr, batch_curr)
                    x_curr = self._apply_tier_global_node(
                        x_curr, batch=batch_curr, tier_idx=tier_idx
                    )

        return x_curr, pos_curr, thickness_embed

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

        x, pos, thickness_embed = self._run_tiered_feature_unet(
            data=data,
            pos_origin=pos_origin,
            pos=pos,
            x=x,
            batch=batch,
            face=face,
            edge_attr=edge_attr,
            thickness=thickness,
            thickness_embed=thickness_embed,
            thickness_paired_idx=thickness_paired_idx,
        )

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
