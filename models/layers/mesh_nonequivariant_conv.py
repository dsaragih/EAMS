import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    act_fn: nn.Module,
    *,
    final_act: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        act_fn,
        nn.Linear(hidden_dim, output_dim),
    ]
    if final_act:
        layers.append(act_fn)
    return nn.Sequential(*layers)


class MeshNonEquivariantConv(nn.Module):
    """Update persistent node, latent-coordinate, edge, and face states."""

    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        edge_nf: int,
        face_nf: int,
        hidden_nf: int | None = None,
        act_fn: nn.Module | None = None,
        residual: bool = True,
        eps: float = 1e-12,
        **ignore_kwargs,
    ):
        super().__init__()
        del ignore_kwargs

        if edge_nf < 1:
            raise ValueError(f"edge_nf must be positive, got: {edge_nf}")
        if face_nf < 1:
            raise ValueError(f"face_nf must be positive, got: {face_nf}")

        self.input_nf = input_nf
        self.output_nf = output_nf
        self.edge_nf = edge_nf
        self.face_nf = face_nf
        self.hidden_nf = hidden_nf or max(input_nf, output_nf, edge_nf, face_nf)
        self.residual = residual
        self.eps = eps
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()

        edge_update_in = edge_nf + 2 * input_nf + 3 + 3 + 1 + 1
        face_update_in = face_nf + 3 * input_nf + 1 + 1 + 3 + 3 + 3 + 3
        node_update_in = input_nf + output_nf + output_nf

        self.edge_mlp = _mlp(
            edge_update_in,
            self.hidden_nf,
            edge_nf,
            self.act_fn,
            final_act=True,
        )
        self.face_mlp = _mlp(
            face_update_in,
            self.hidden_nf,
            face_nf,
            self.act_fn,
            final_act=True,
        )
        self.edge_to_node_mlp = _mlp(
            edge_nf, self.hidden_nf, output_nf, self.act_fn, final_act=True
        )
        self.face_to_node_mlp = _mlp(
            face_nf, self.hidden_nf, output_nf, self.act_fn, final_act=True
        )
        self.node_mlp = _mlp(node_update_in, self.hidden_nf, output_nf, self.act_fn)
        self.edge_u_gate = _mlp(edge_nf, self.hidden_nf, 1, self.act_fn)
        self.face_u_gate = _mlp(face_nf, self.hidden_nf, 1, self.act_fn)
        self.gamma = nn.Parameter(torch.zeros(()))

        self.residual_mlp = (
            nn.Linear(input_nf, output_nf, bias=False) if input_nf != output_nf else None
        )

    def _edge_terms(
        self,
        *,
        z: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center, neighbor = edge_index[0].long(), edge_index[1].long()
        r0 = x[neighbor] - x[center]
        rl = u[neighbor] - u[center]
        d0 = r0.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        dl = rl.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        edge_input = torch.cat(
            [edge_state, z[center], z[neighbor], r0, rl, d0, dl], dim=-1
        )
        edge_state = self.edge_mlp(edge_input)
        edge_node_msg = self.edge_to_node_mlp(edge_state)
        edge_node_agg = scatter(
            edge_node_msg, center, dim=0, dim_size=z.shape[0], reduce="sum"
        )
        edge_u = rl * torch.tanh(self.edge_u_gate(edge_state))
        edge_u_agg = scatter(edge_u, center, dim=0, dim_size=z.shape[0], reduce="sum")
        return edge_state, edge_node_agg, edge_u_agg

    def _face_geometry(
        self, coord: torch.Tensor, face: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, j, k = face[0].long(), face[1].long(), face[2].long()
        r_ij = coord[j] - coord[i]
        r_jk = coord[k] - coord[j]
        normal = torch.cross(r_ij, r_jk, dim=1)
        area = normal.norm(dim=-1, keepdim=True) / 2
        centroid = (coord[i] + coord[j] + coord[k]) / 3
        return normal, area, centroid

    def _face_terms(
        self,
        *,
        z: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        face: torch.Tensor,
        face_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if face.numel() == 0:
            face_state = self.face_mlp(
                face_state.new_empty((0, self.face_mlp[0].in_features))
            )
            return (
                face_state,
                z.new_zeros((z.shape[0], self.output_nf)),
                u.new_zeros((u.shape[0], 3)),
            )

        i, j, k = face[0].long(), face[1].long(), face[2].long()
        normal_0, area_0, centroid_0 = self._face_geometry(x, face)
        normal_l, area_l, centroid_l = self._face_geometry(u, face)
        face_input = torch.cat(
            [
                face_state,
                z[i],
                z[j],
                z[k],
                area_0,
                area_l,
                normal_0,
                normal_l,
                centroid_0,
                centroid_l,
            ],
            dim=-1,
        )
        face_state = self.face_mlp(face_input)
        face_node_msg = self.face_to_node_mlp(face_state)
        face_index = face.T.reshape(-1).long()
        face_node_agg = scatter(
            face_node_msg.repeat_interleave(3, dim=0),
            face_index,
            dim=0,
            dim_size=z.shape[0],
            reduce="sum",
        )
        face_u = normal_l * torch.tanh(self.face_u_gate(face_state))
        face_u_agg = scatter(
            face_u.repeat_interleave(3, dim=0),
            face_index,
            dim=0,
            dim_size=u.shape[0],
            reduce="sum",
        )
        return face_state, face_node_agg, face_u_agg

    def forward(
        self,
        *,
        z: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        face: torch.Tensor,
        face_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_state, edge_node_agg, edge_u_agg = self._edge_terms(
            z=z, x=x, u=u, edge_index=edge_index, edge_state=edge_state
        )
        face_state, face_node_agg, face_u_agg = self._face_terms(
            z=z, x=x, u=u, face=face, face_state=face_state
        )
        z_update = self.node_mlp(torch.cat([z, edge_node_agg, face_node_agg], dim=-1))

        if self.residual:
            residual = self.residual_mlp(z) if self.residual_mlp is not None else z
            z_update = residual + z_update

        u = u + self.gamma * (edge_u_agg + face_u_agg)
        return z_update, u, edge_state, face_state
