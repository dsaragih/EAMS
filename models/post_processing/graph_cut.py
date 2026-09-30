import numpy as np
import torch
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from util.losses import get_adjacent_edge_pairs, get_adjacent_vertex_pairs

from .base import BasePostProcessor, PostProcessInput, PostProcessOutput


class GraphCutPostProcessor(BasePostProcessor):
    def __init__(
        self,
        enabled: bool = True,
        run_on_validation: bool = True,
        run_on_test: bool = True,
        smoothness_weight: float = 1.0,
        edge_smoothness_weight: float | None = None,
        geometry_weight: float = 0.0,
        geometry_sigma: float = 1.0,
        logit_temperature: float = 1.0,
        include_background: bool = True,
        class_ids: list[int] | None = None,
        capacity_scale: int = 1000,
    ):
        super().__init__(
            enabled=enabled,
            run_on_validation=run_on_validation,
            run_on_test=run_on_test,
        )
        self.smoothness_weight = float(smoothness_weight)
        self.edge_smoothness_weight = (
            float(edge_smoothness_weight)
            if edge_smoothness_weight is not None
            else float(smoothness_weight)
        )
        self.geometry_weight = float(geometry_weight)
        self.geometry_sigma = max(float(geometry_sigma), 1e-8)
        self.logit_temperature = max(float(logit_temperature), 1e-8)
        self.include_background = bool(include_background)
        self.class_ids = class_ids
        self.capacity_scale = max(int(capacity_scale), 1)

    def process(self, data: PostProcessInput) -> PostProcessOutput:
        if data.logits is None:
            raise ValueError("GraphCutPostProcessor requires logits")
        if data.logits.ndim != 2:
            raise ValueError(
                f"Expected logits with shape (N, C), got {tuple(data.logits.shape)}"
            )
        if data.label_kind not in {"vertex", "edge"}:
            raise ValueError(f"Unsupported label kind: {data.label_kind}")

        logits = data.logits.float()
        labels = data.labels.long()
        num_items, num_classes = logits.shape
        if labels.numel() != num_items:
            raise ValueError(
                f"Labels length {labels.numel()} does not match logits length {num_items}"
            )

        valid_mask = labels != -1
        if not valid_mask.any() or num_items == 0:
            return PostProcessOutput(labels=labels)

        probs = torch.softmax(logits / self.logit_temperature, dim=1)
        adjacency_pairs, pairwise_weights = self._build_pairwise_terms(data)

        active_classes = self._resolve_class_ids(num_classes)
        active_masks = torch.zeros(
            (num_items, num_classes), dtype=torch.bool, device=labels.device
        )

        for class_id in active_classes:
            class_prob = probs[:, class_id].clamp(min=1e-6, max=1.0 - 1e-6)
            foreground_cost = -torch.log(class_prob)
            background_cost = -torch.log((1.0 - class_prob).clamp(min=1e-6))
            cut_mask = self._solve_binary_cut(
                background_cost=background_cost,
                foreground_cost=foreground_cost,
                adjacency_pairs=adjacency_pairs,
                pairwise_weights=pairwise_weights,
            )
            active_masks[:, class_id] = cut_mask & valid_mask

        refined_labels = labels.clone()
        any_active = active_masks.any(dim=1) & valid_mask
        if any_active.any():
            candidate_scores = probs.clone()
            candidate_scores[~active_masks] = -1.0
            best_classes = candidate_scores.argmax(dim=1)
            refined_labels[any_active] = best_classes[any_active]

        return PostProcessOutput(
            labels=refined_labels,
            metadata={
                "num_refined": int(
                    ((refined_labels != labels) & valid_mask)
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
            },
        )

    def _resolve_class_ids(self, num_classes: int) -> list[int]:
        if self.class_ids is not None:
            return [
                class_id for class_id in self.class_ids if 0 <= class_id < num_classes
            ]
        start_idx = 0 if self.include_background else 1
        return list(range(start_idx, num_classes))

    def _build_pairwise_terms(
        self,
        data: PostProcessInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mesh = data.mesh
        if not hasattr(mesh, "edge_index"):
            empty_pairs = torch.empty(
                (2, 0), dtype=torch.long, device=data.labels.device
            )
            empty_weights = torch.empty(
                (0,), dtype=torch.float32, device=data.labels.device
            )
            return empty_pairs, empty_weights

        edge_index = mesh.edge_index.long().to(data.labels.device)
        if data.label_kind == "vertex":
            adjacency_pairs = get_adjacent_vertex_pairs(edge_index=edge_index)
            base_weight = self.smoothness_weight
            geometry_points = (
                mesh.pos.to(data.labels.device) if hasattr(mesh, "pos") else None
            )
        else:
            if not hasattr(mesh, "pos"):
                raise ValueError(
                    "Edge graph cut requires mesh.pos for adjacency geometry"
                )
            adjacency_pairs = get_adjacent_edge_pairs(
                edges=edge_index,
                num_vertices=int(mesh.pos.shape[0]),
                edge_labels=None,
            )
            base_weight = self.edge_smoothness_weight
            geometry_points = None

        if adjacency_pairs.numel() == 0:
            empty_weights = torch.empty(
                (0,), dtype=torch.float32, device=data.labels.device
            )
            return adjacency_pairs.to(data.labels.device), empty_weights

        pairwise_weights = torch.full(
            (adjacency_pairs.shape[1],),
            fill_value=base_weight,
            dtype=torch.float32,
            device=data.labels.device,
        )
        if self.geometry_weight <= 0.0:
            return adjacency_pairs.to(data.labels.device), pairwise_weights

        if data.label_kind == "vertex" and geometry_points is not None:
            left = geometry_points[adjacency_pairs[0]]
            right = geometry_points[adjacency_pairs[1]]
        else:
            edge_midpoints = (mesh.pos[edge_index[0]] + mesh.pos[edge_index[1]]) / 2.0
            edge_midpoints = edge_midpoints.to(data.labels.device)
            left = edge_midpoints[adjacency_pairs[0]]
            right = edge_midpoints[adjacency_pairs[1]]

        squared_distance = (left - right).pow(2).sum(dim=1)
        geometry_scale = torch.exp(
            -squared_distance / (2.0 * self.geometry_sigma * self.geometry_sigma)
        )
        pairwise_weights = pairwise_weights * (
            1.0 + self.geometry_weight * geometry_scale
        )
        return adjacency_pairs.to(data.labels.device), pairwise_weights

    def _solve_binary_cut(
        self,
        *,
        background_cost: torch.Tensor,
        foreground_cost: torch.Tensor,
        adjacency_pairs: torch.Tensor,
        pairwise_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(background_cost.numel())
        source = num_nodes
        sink = num_nodes + 1
        capacities = self._build_capacity_matrix(
            background_cost=background_cost,
            foreground_cost=foreground_cost,
            adjacency_pairs=adjacency_pairs,
            pairwise_weights=pairwise_weights,
            source=source,
            sink=sink,
        )
        flow_result = maximum_flow(capacities, source, sink)
        residual = capacities - flow_result.flow
        residual.data = np.clip(residual.data, a_min=0, a_max=None)
        residual.eliminate_zeros()

        reachable = breadth_first_order(
            residual, source, directed=True, return_predecessors=False
        )
        source_side = np.zeros(num_nodes + 2, dtype=bool)
        source_side[reachable] = True
        return torch.from_numpy(source_side[:num_nodes]).to(
            device=background_cost.device, dtype=torch.bool
        )

    def _build_capacity_matrix(
        self,
        *,
        background_cost: torch.Tensor,
        foreground_cost: torch.Tensor,
        adjacency_pairs: torch.Tensor,
        pairwise_weights: torch.Tensor,
        source: int,
        sink: int,
    ) -> csr_array:
        num_nodes = int(background_cost.numel())
        row_parts = [
            torch.full(
                (num_nodes,), source, dtype=torch.long, device=background_cost.device
            ),
            torch.arange(num_nodes, device=background_cost.device, dtype=torch.long),
        ]
        col_parts = [
            torch.arange(num_nodes, device=background_cost.device, dtype=torch.long),
            torch.full(
                (num_nodes,), sink, dtype=torch.long, device=background_cost.device
            ),
        ]
        cap_parts = [background_cost, foreground_cost]

        if adjacency_pairs.numel() > 0:
            valid_pairs = adjacency_pairs[0] != adjacency_pairs[1]
            if valid_pairs.any():
                filtered_pairs = adjacency_pairs[:, valid_pairs]
                filtered_weights = pairwise_weights[valid_pairs]
                row_parts.append(filtered_pairs[0])
                row_parts.append(filtered_pairs[1])
                col_parts.append(filtered_pairs[1])
                col_parts.append(filtered_pairs[0])
                cap_parts.append(filtered_weights)
                cap_parts.append(filtered_weights)

        rows = torch.cat(row_parts, dim=0)
        cols = torch.cat(col_parts, dim=0)
        capacities = torch.cat(cap_parts, dim=0)
        capacities = self._quantize_capacities(capacities)

        return csr_array(
            (capacities, (rows.detach().cpu().numpy(), cols.detach().cpu().numpy())),
            shape=(num_nodes + 2, num_nodes + 2),
            dtype=np.int64,
        )

    def _quantize_capacities(self, capacities: torch.Tensor) -> np.ndarray:
        scaled = torch.round(capacities.clamp_min(0.0) * float(self.capacity_scale))
        positive_mask = capacities > 0
        scaled[(scaled == 0) & positive_mask] = 1
        return scaled.to(dtype=torch.int64).detach().cpu().numpy()
