import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from models.utils import *

from loguru import logger


class MeshEGNN(MessagePassing):
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
        **ignore_kwargs,
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

        self.edge_mlp_area_mini = nn.Sequential(
            nn.Linear(input_nf * 2 + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.edge_mlp_area = nn.Sequential(
            nn.Linear(input_nf * 2 + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + input_nf, hidden_nf),
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

        layer_ = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer_.weight, gain=0.001)
        coord_area = []
        coord_area.append(nn.Linear(hidden_nf, hidden_nf))
        coord_area.append(act_fn)
        coord_area.append(layer_)
        if self.tanh:
            coord_area.append(nn.Tanh())
        self.coord_area = nn.Sequential(*coord_area)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # Optional thickness embedding update (enabled when forward() receives
        # thickness_embed + thickness_paired_idx).
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
        face,
        thickness_embed=None,
        thickness_paired_idx=None,
        thickness_scale: torch.Tensor | None = None,
    ):
        radial, coord_diff = self.coord2radial(edge_index, coord)

        out = self.propagate(
            edge_index, x=x, radial=radial, edge_attr=edge_attr, coord_diff=coord_diff
        )
        out_coord, out_message = out

        m_ijk = []
        muls = []
        for i in range(3):
            vec1 = coord[face[i % 3]] - coord[face[(i + 1) % 3]]
            vec2 = coord[face[i % 3]] - coord[face[(i + 2) % 3]]
            normal_m = torch.cross(vec1, vec2, dim=1)
            area_m = torch.norm(normal_m, p=2, dim=1) / 2
            m = self.edge_mlp_area(
                torch.cat(
                    [
                        x[face[i % 3]],
                        0.5 * (x[face[(i + 1) % 3]] + x[face[(i + 2) % 3]]),
                        area_m.unsqueeze(-1),
                    ],
                    dim=-1,
                ).float()
            )
            mul = normal_m * self.coord_area(m)

            m_ijk.append(m)
            muls.append(mul)

        m_ijk = torch.cat(m_ijk, dim=0)
        muls = torch.cat(muls, dim=0)
        index = face.T.flatten()

        area_coord = scatter(muls, index, dim=0, dim_size=x.shape[0])

        # coord = posi + out_coord
        coord = coord + out_coord + area_coord

        # xi = \phi([xi, out_message])
        out = self.node_mlp(
            torch.cat(
                [x, out_message, scatter(m_ijk, index, dim=0, dim_size=x.shape[0])],
                dim=1,
            )
        )

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


class MeshEGNNmc(MessagePassing):
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
        **ignore_kwargs,
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

        self.edge_mlp_area = nn.Sequential(
            nn.Linear(input_nf * 2 + num_vectors_in, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + input_nf, hidden_nf),
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

        layer_ = nn.Linear(hidden_nf, num_vectors_in * num_vectors_out, bias=False)
        torch.nn.init.xavier_uniform_(layer_.weight, gain=0.001)
        coord_area = []
        coord_area.append(nn.Linear(hidden_nf, hidden_nf))
        coord_area.append(act_fn)
        coord_area.append(layer_)
        if self.tanh:
            coord_area.append(nn.Tanh())
        self.coord_area = nn.Sequential(*coord_area)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        if (
            self.num_vectors_in != 1
            and self.num_vectors_out != 1
            and self.num_vectors_in != self.num_vectors_out
        ):
            self.project = nn.Linear(num_vectors_in, num_vectors_out)
        else:
            self.project = None

        # Optional thickness embedding update (enabled when forward() receives
        # thickness_embed + thickness_paired_idx).
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
        face,
        thickness_embed=None,
        thickness_paired_idx=None,
        thickness_scale: torch.Tensor | None = None,
    ):
        radial, coord_diff = self.coord2radial(edge_index, coord)

        out = self.propagate(
            edge_index, x=x, radial=radial, edge_attr=edge_attr, coord_diff=coord_diff
        )
        out_coord, out_message = out

        if coord_diff.dim() == 2:
            coord = coord.unsqueeze(2).repeat(1, 1, self.num_vectors_out)

        m_ijk = []
        muls = []
        for i in range(3):
            vec1 = coord[face[i % 3]] - coord[face[(i + 1) % 3]]
            vec2 = coord[face[i % 3]] - coord[face[(i + 2) % 3]]
            normal_m = torch.cross(vec1, vec2, dim=1)
            area_m = torch.norm(normal_m, p=2, dim=1) / 2
            if area_m.dim() == 1:
                area_m = area_m.unsqueeze(-1)

            m = self.edge_mlp_area(
                torch.cat(
                    [
                        x[face[i % 3]],
                        0.5 * (x[face[(i + 1) % 3]] + x[face[(i + 2) % 3]]),
                        area_m,
                    ],
                    dim=-1,
                ).float()
            )

            if normal_m.dim() == 2:
                normal_m = normal_m.unsqueeze(-1)
            mul = torch.einsum(
                "bij,bci->bcj",
                self.coord_area(m).view(-1, self.num_vectors_in, self.num_vectors_out),
                normal_m,
            )

            m_ijk.append(m)
            muls.append(mul)

        m_ijk = torch.cat(m_ijk, dim=0)
        muls = torch.cat(muls, dim=0)
        index = face.T.flatten()

        area_coord = scatter(muls, index, dim=0, dim_size=x.shape[0])

        if self.project is not None:
            coord = self.project(coord)

        if self.last_layer:
            coord = coord.mean(dim=2, keepdim=True) + out_coord
        else:
            coord += out_coord

        coord += area_coord

        out = self.node_mlp(
            torch.cat(
                [x, out_message, scatter(m_ijk, index, dim=0, dim_size=x.shape[0])],
                dim=1,
            )
        )

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


class MeshEGNNFastVirtual(MeshEGNN):
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
        num_virtual_nodes: int = 8,
        virtual_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            input_nf=input_nf,
            output_nf=output_nf,
            hidden_nf=hidden_nf,
            edges_in_d=edges_in_d,
            act_fn=act_fn,
            residual=residual,
            attention=attention,
            normalize=normalize,
            coords_agg=coords_agg,
            tanh=tanh,
            **kwargs,
        )
        self.input_nf = input_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.num_virtual_nodes = num_virtual_nodes

        self.virtual_edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + num_virtual_nodes, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 3 + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        self.virtual_update_mlp = nn.Sequential(
            nn.Linear(input_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Dropout(virtual_dropout),
            nn.Linear(hidden_nf, output_nf),
        )
        self.virtual_residual_mlp = (
            nn.Linear(input_nf, output_nf, bias=False)
            if input_nf != output_nf
            else None
        )

        self.virtual_coord_mlp_node = self._build_scalar_coord_mlp(hidden_nf, act_fn)
        self.virtual_coord_mlp_virtual = self._build_scalar_coord_mlp(hidden_nf, act_fn)

    def _build_scalar_coord_mlp(self, hidden_nf, act_fn):
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        return nn.Sequential(*coord_mlp)

    def _compute_face_terms(self, x, coord, face):
        m_ijk = []
        muls = []
        for i in range(3):
            vec1 = coord[face[i % 3]] - coord[face[(i + 1) % 3]]
            vec2 = coord[face[i % 3]] - coord[face[(i + 2) % 3]]
            normal_m = torch.cross(vec1, vec2, dim=1)
            area_m = torch.norm(normal_m, p=2, dim=1) / 2
            m = self.edge_mlp_area(
                torch.cat(
                    [
                        x[face[i % 3]],
                        0.5 * (x[face[(i + 1) % 3]] + x[face[(i + 2) % 3]]),
                        area_m.unsqueeze(-1),
                    ],
                    dim=-1,
                ).float()
            )
            mul = normal_m * self.coord_area(m)
            m_ijk.append(m)
            muls.append(mul)

        m_ijk = torch.cat(m_ijk, dim=0)
        muls = torch.cat(muls, dim=0)
        index = face.T.flatten()
        face_message_agg = scatter(m_ijk, index, dim=0, dim_size=x.shape[0])
        area_coord = scatter(muls, index, dim=0, dim_size=x.shape[0])
        return face_message_agg, area_coord

    def _compute_virtual_terms(
        self,
        x,
        coord,
        virtual_node_feat,
        virtual_node_coord,
        batch,
    ):
        coord_mean = scatter(
            coord, batch, dim=0, dim_size=virtual_node_feat.shape[0], reduce="mean"
        )
        centered_virtual = virtual_node_coord - coord_mean.unsqueeze(1)
        virtual_corr = torch.einsum("bvc,bwc->bvw", centered_virtual, centered_virtual)
        coord_diff = virtual_node_coord[batch] - coord.unsqueeze(1)
        radial = coord_diff.square().sum(dim=-1, keepdim=True)
        virtual_inputs = torch.cat(
            [
                x.unsqueeze(1).expand(-1, self.num_virtual_nodes, -1),
                virtual_node_feat[batch],
                radial,
                virtual_corr[batch],
            ],
            dim=-1,
        )
        virtual_messages = self.virtual_edge_mlp(virtual_inputs)

        virtual_message_agg = virtual_messages.mean(dim=1)
        virtual_coord_term = (
            -coord_diff * self.virtual_coord_mlp_node(virtual_messages)
        ).mean(dim=1)

        num_graphs = virtual_node_feat.shape[0]
        graph_virtual_messages = scatter(
            virtual_messages.reshape(
                x.shape[0], self.num_virtual_nodes * self.hidden_nf
            ),
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce="mean",
        ).reshape(num_graphs, self.num_virtual_nodes, self.hidden_nf)

        virtual_feat_update = self.virtual_update_mlp(
            torch.cat([virtual_node_feat, graph_virtual_messages], dim=-1)
        )
        if self.virtual_residual_mlp is not None:
            virtual_node_feat = self.act(
                self.virtual_residual_mlp(virtual_node_feat) + virtual_feat_update
            )
        else:
            virtual_node_feat = virtual_node_feat + virtual_feat_update

        virtual_coord_delta = coord_diff * self.virtual_coord_mlp_virtual(
            virtual_messages
        )
        virtual_coord_delta = scatter(
            virtual_coord_delta.reshape(
                x.shape[0], self.num_virtual_nodes * coord.shape[1]
            ),
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce="mean",
        ).reshape(num_graphs, self.num_virtual_nodes, coord.shape[1])
        virtual_node_coord = virtual_node_coord + virtual_coord_delta

        return (
            virtual_message_agg,
            virtual_coord_term,
            virtual_node_feat,
            virtual_node_coord,
        )

    def forward(
        self,
        x,
        edge_index,
        coord,
        edge_attr,
        face,
        batch,
        virtual_node_feat,
        virtual_node_coord,
        thickness_embed=None,
        thickness_paired_idx=None,
        thickness_scale: torch.Tensor | None = None,
    ):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        out_coord, out_message = self.propagate(
            edge_index, x=x, radial=radial, edge_attr=edge_attr, coord_diff=coord_diff
        )
        face_message_agg, area_coord = self._compute_face_terms(x, coord, face)
        (
            virtual_message_agg,
            virtual_coord_term,
            virtual_node_feat,
            virtual_node_coord,
        ) = self._compute_virtual_terms(
            x, coord, virtual_node_feat, virtual_node_coord, batch
        )

        coord = coord + out_coord + area_coord + virtual_coord_term
        out = self.node_mlp(
            torch.cat([x, out_message, face_message_agg, virtual_message_agg], dim=1)
        )

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
            return out, coord, virtual_node_feat, virtual_node_coord, thickness_embed

        return out, coord, virtual_node_feat, virtual_node_coord


class MeshEGNNmcFastVirtual(MeshEGNNmc):
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
        num_virtual_nodes: int = 8,
        virtual_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            input_nf=input_nf,
            output_nf=output_nf,
            hidden_nf=hidden_nf,
            edges_in_d=edges_in_d,
            act_fn=act_fn,
            residual=residual,
            attention=attention,
            normalize=normalize,
            coords_agg=coords_agg,
            tanh=tanh,
            num_vectors_in=num_vectors_in,
            num_vectors_out=num_vectors_out,
            last_layer=last_layer,
            **kwargs,
        )
        self.input_nf = input_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.num_virtual_nodes = num_virtual_nodes

        self.virtual_edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + num_vectors_in + num_virtual_nodes, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 3 + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        self.virtual_update_mlp = nn.Sequential(
            nn.Linear(input_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Dropout(virtual_dropout),
            nn.Linear(hidden_nf, output_nf),
        )
        self.virtual_residual_mlp = (
            nn.Linear(input_nf, output_nf, bias=False)
            if input_nf != output_nf
            else None
        )
        self.virtual_coord_mlp_node = self._build_matrix_coord_mlp(hidden_nf, act_fn)
        self.virtual_coord_mlp_virtual = self._build_matrix_coord_mlp(hidden_nf, act_fn)

    def _build_matrix_coord_mlp(self, hidden_nf, act_fn):
        layer = nn.Linear(
            hidden_nf, self.num_vectors_in * self.num_vectors_out, bias=False
        )
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        return nn.Sequential(*coord_mlp)

    def _compute_face_terms(self, x, coord, face):
        m_ijk = []
        muls = []
        for i in range(3):
            vec1 = coord[face[i % 3]] - coord[face[(i + 1) % 3]]
            vec2 = coord[face[i % 3]] - coord[face[(i + 2) % 3]]
            normal_m = torch.cross(vec1, vec2, dim=1)
            area_m = torch.norm(normal_m, p=2, dim=1) / 2
            if area_m.dim() == 1:
                area_m = area_m.unsqueeze(-1)

            m = self.edge_mlp_area(
                torch.cat(
                    [
                        x[face[i % 3]],
                        0.5 * (x[face[(i + 1) % 3]] + x[face[(i + 2) % 3]]),
                        area_m,
                    ],
                    dim=-1,
                ).float()
            )

            if normal_m.dim() == 2:
                normal_m = normal_m.unsqueeze(-1)
            mul = torch.einsum(
                "bij,bci->bcj",
                self.coord_area(m).view(-1, self.num_vectors_in, self.num_vectors_out),
                normal_m,
            )
            m_ijk.append(m)
            muls.append(mul)

        m_ijk = torch.cat(m_ijk, dim=0)
        muls = torch.cat(muls, dim=0)
        index = face.T.flatten()
        face_message_agg = scatter(m_ijk, index, dim=0, dim_size=x.shape[0])
        area_coord = scatter(muls, index, dim=0, dim_size=x.shape[0])
        return face_message_agg, area_coord

    def _compute_virtual_terms(
        self,
        x,
        coord,
        virtual_node_feat,
        virtual_node_coord,
        batch,
    ):
        coord_mean = scatter(
            coord, batch, dim=0, dim_size=virtual_node_feat.shape[0], reduce="mean"
        )
        centered_virtual = virtual_node_coord - coord_mean.unsqueeze(1)
        virtual_corr = torch.einsum(
            "bvci,bwci->bvw", centered_virtual, centered_virtual
        )
        coord_diff = virtual_node_coord[batch] - coord.unsqueeze(1)
        radial = coord_diff.square().sum(dim=2)
        virtual_inputs = torch.cat(
            [
                x.unsqueeze(1).expand(-1, self.num_virtual_nodes, -1),
                virtual_node_feat[batch],
                radial,
                virtual_corr[batch],
            ],
            dim=-1,
        )
        virtual_messages = self.virtual_edge_mlp(virtual_inputs)
        virtual_message_agg = virtual_messages.mean(dim=1)

        node_matrix = self.virtual_coord_mlp_node(virtual_messages).view(
            -1,
            self.num_virtual_nodes,
            self.num_vectors_in,
            self.num_vectors_out,
        )
        virtual_coord_term = torch.einsum(
            "nvij,nvci->nvcj", node_matrix, -coord_diff
        ).mean(dim=1)

        num_graphs = virtual_node_feat.shape[0]
        graph_virtual_messages = scatter(
            virtual_messages.reshape(
                x.shape[0], self.num_virtual_nodes * self.hidden_nf
            ),
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce="mean",
        ).reshape(num_graphs, self.num_virtual_nodes, self.hidden_nf)

        virtual_feat_update = self.virtual_update_mlp(
            torch.cat([virtual_node_feat, graph_virtual_messages], dim=-1)
        )
        if self.virtual_residual_mlp is not None:
            virtual_node_feat = self.act(
                self.virtual_residual_mlp(virtual_node_feat) + virtual_feat_update
            )
        else:
            virtual_node_feat = virtual_node_feat + virtual_feat_update

        virtual_matrix = self.virtual_coord_mlp_virtual(virtual_messages).view(
            -1,
            self.num_virtual_nodes,
            self.num_vectors_in,
            self.num_vectors_out,
        )
        virtual_coord_delta = torch.einsum(
            "nvij,nvci->nvcj", virtual_matrix, coord_diff
        )
        virtual_coord_delta = scatter(
            virtual_coord_delta.reshape(
                x.shape[0],
                self.num_virtual_nodes * coord.shape[1] * self.num_vectors_out,
            ),
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce="mean",
        ).reshape(
            num_graphs,
            self.num_virtual_nodes,
            coord.shape[1],
            self.num_vectors_out,
        )

        return (
            virtual_message_agg,
            virtual_coord_term,
            virtual_node_feat,
            virtual_coord_delta,
        )

    def forward(
        self,
        x,
        edge_index,
        coord,
        edge_attr,
        face,
        batch,
        virtual_node_feat,
        virtual_node_coord,
        thickness_embed=None,
        thickness_paired_idx=None,
        thickness_scale: torch.Tensor | None = None,
    ):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        out_coord, out_message = self.propagate(
            edge_index, x=x, radial=radial, edge_attr=edge_attr, coord_diff=coord_diff
        )

        if coord_diff.dim() == 2:
            coord = coord.unsqueeze(2).repeat(1, 1, self.num_vectors_out)

        face_message_agg, area_coord = self._compute_face_terms(x, coord, face)
        (
            virtual_message_agg,
            virtual_coord_term,
            virtual_node_feat,
            virtual_coord_delta,
        ) = self._compute_virtual_terms(
            x, coord, virtual_node_feat, virtual_node_coord, batch
        )

        if self.project is not None:
            coord = self.project(coord)
            virtual_node_coord = self.project(virtual_node_coord)

        if self.last_layer:
            coord = coord.mean(dim=2, keepdim=True) + out_coord
            virtual_node_coord = virtual_node_coord.mean(dim=3, keepdim=True)
        else:
            coord = coord + out_coord

        coord = coord + area_coord + virtual_coord_term
        virtual_node_coord = virtual_node_coord + virtual_coord_delta

        out = self.node_mlp(
            torch.cat([x, out_message, face_message_agg, virtual_message_agg], dim=1)
        )

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
            return out, coord, virtual_node_feat, virtual_node_coord, thickness_embed

        return out, coord, virtual_node_feat, virtual_node_coord
