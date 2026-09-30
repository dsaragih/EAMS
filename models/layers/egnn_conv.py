import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from models.utils import *

from loguru import logger


class EGNN(MessagePassing):
    """Different from the above since it separates the edge assignment
    from the computation (this allows for great reduction in time and
    computations when the graph is locally or sparse connected).
    * aggr: one of ["add", "mean", "max"]
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
        **ignore_kwargs
    ):
        super().__init__(aggr="add")
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        if input_nf != output_nf:
            self.residual_mlp = nn.Linear(input_nf, output_nf, bias=False)
            self.act = act_fn
        else:
            self.residual_mlp = None

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        self.thickness_update_mlp = nn.Sequential(
            nn.Linear(input_nf + 2 * output_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        self.node_thickness_mlp = nn.Sequential(
            nn.Linear(2 * output_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(
        self,
        x,
        edge_index,
        coord,
        edge_attr,
        face=None,
        thickness_embed=None,
        thickness_paired_idx=None,
        thickness_scale: torch.Tensor | None = None,
    ):
        # coors, feats = x[:, :3], x[:, 3:]
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # out_message = mi = \sum mij, where mij \phi([xi, xj, ||posi-posj||, aij]) ; out_coord = C \sum (posi - posj) \phi (mij))
        out = self.propagate(
            edge_index, x=x, radial=radial, edge_attr=edge_attr, coord_diff=coord_diff
        )
        out_coord, out_message = out[:, :3], out[:, 3:]

        # coord = posi + out_coord
        coord = coord + out_coord

        # xi = \phi([xi, out_message])
        out = self.node_mlp(torch.cat([x, out_message], dim=1))

        if self.residual:
            if self.residual_mlp is not None:
                out = self.act(self.residual_mlp(x) + out)
            else:
                out = x + out

        if (thickness_embed is None) ^ (thickness_paired_idx is None):
            raise ValueError(
                "thickness_embed and thickness_paired_idx must be provided together"
            )

        if thickness_embed is not None:
            assert (
                thickness_scale is not None
            ), "thickness_scale must be provided when thickness_embed is used"
            paired = thickness_paired_idx.long()
            out_paired = out[paired]
            thickness_embed = thickness_scale * self.thickness_update_mlp(
                torch.cat([thickness_embed, out, out_paired], dim=-1)
            )
            out = self.node_thickness_mlp(torch.cat([out, thickness_embed], dim=-1))
            return out, coord, thickness_embed

        return out, coord

    def message(self, x_i, x_j, radial, edge_attr, coord_diff):
        if edge_attr is None:  # Unused.
            m_ij = torch.cat([x_i, x_j, radial], dim=1)
        else:
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            m_ij = torch.cat([x_i, x_j, radial, edge_attr], dim=1)

        m_ij = self.edge_mlp(m_ij)

        if self.attention:
            att_val = self.att_mlp(m_ij)
            m_ij = m_ij * att_val

        c_ij = coord_diff * self.coord_mlp(m_ij)
        return torch.cat([c_ij, m_ij], dim=-1)


class EGNNmc(MessagePassing):
    """Different from the above since it separates the edge assignment
    from the computation (this allows for great reduction in time and
    computations when the graph is locally or sparse connected).
    * aggr: one of ["add", "mean", "max"]
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
        num_vectors_in=1,
        num_vectors_out=1,
        last_layer=False,
        **ignore_kwargs
    ):
        super().__init__(aggr="add")
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.num_vectors_in = num_vectors_in
        self.num_vectors_out = num_vectors_out
        self.last_layer = last_layer
        self.epsilon = 1e-8
        edge_coords_nf = 1

        if input_nf != output_nf:
            self.residual_mlp = nn.Linear(input_nf, output_nf, bias=False)
            self.act = act_fn
        else:
            self.residual_mlp = None

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + num_vectors_in + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, num_vectors_in * num_vectors_out, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        self.thickness_update_mlp = nn.Sequential(
            nn.Linear(input_nf + 2 * output_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        self.node_thickness_mlp = nn.Sequential(
            nn.Linear(2 * output_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        if radial.dim() == 3:
            radial = radial.squeeze(1)

        return radial, coord_diff

    def forward(
        self,
        x,
        edge_index,
        coord,
        edge_attr,
        face=None,
        thickness_embed=None,
        thickness_paired_idx=None,
        thickness_scale: torch.Tensor | None = None,
    ):
        # coors, feats = x[:, :3], x[:, 3:]
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # out_message = mi = \sum mij, where mij \phi([xi, xj, ||posi-posj||, aij]) ; out_coord = C \sum (posi - posj) \phi (mij))
        out = self.propagate(
            edge_index, x=x, radial=radial, edge_attr=edge_attr, coord_diff=coord_diff
        )
        out_coord, out_message = out

        if coord_diff.dim() == 2:
            coord = coord.unsqueeze(2).repeat(1, 1, self.num_vectors_out)

        if self.last_layer:
            coord = coord.mean(dim=2, keepdim=True) + out_coord
        else:
            coord += out_coord

        out = self.node_mlp(torch.cat([x, out_message], dim=1))

        if self.residual:
            if self.residual_mlp is not None:
                out = self.act(self.residual_mlp(x) + out)
            else:
                out = x + out

        if (thickness_embed is None) ^ (thickness_paired_idx is None):
            raise ValueError(
                "thickness_embed and thickness_paired_idx must be provided together"
            )

        if thickness_embed is not None:
            assert (
                thickness_scale is not None
            ), "thickness_scale must be provided when thickness_embed is used"
            paired = thickness_paired_idx.long()
            out_paired = out[paired]
            thickness_embed = thickness_scale * self.thickness_update_mlp(
                torch.cat([thickness_embed, out, out_paired], dim=-1)
            )
            out = self.node_thickness_mlp(torch.cat([out, thickness_embed], dim=-1))
            return out, coord, thickness_embed

        return out, coord

    def message(self, x_i, x_j, radial, edge_attr, coord_diff):
        if edge_attr is None:  # Unused.
            m_ij = torch.cat([x_i, x_j, radial], dim=1)
        else:
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            m_ij = torch.cat([x_i, x_j, radial, edge_attr], dim=1)

        m_ij = self.edge_mlp(m_ij)

        if self.attention:
            att_val = self.att_mlp(m_ij)
            m_ij = m_ij * att_val

        coord_matrix = self.coord_mlp(m_ij).view(
            -1, self.num_vectors_in, self.num_vectors_out
        )

        if coord_diff.dim() == 2:
            coord_diff = coord_diff.unsqueeze(2)

        c_ij = torch.einsum("bij,bci->bcj", coord_matrix, coord_diff)
        return c_ij, m_ij

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        c = self.aggr_module(inputs[0], index, ptr=ptr, dim_size=dim_size, dim=0)
        m = self.aggr_module(
            inputs[1], index, ptr=ptr, dim_size=dim_size, dim=self.node_dim
        )
        return c, m
