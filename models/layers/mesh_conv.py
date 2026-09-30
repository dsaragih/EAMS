import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, num_points):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(num_points, num_points//4),
            nn.ReLU(),
            nn.Linear(num_points//4, num_points),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, x.size(1))
        return x

class CBAMModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, num_points=1024):
        super(CBAMModule, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        # self.spatial_attention = SpatialAttentionModule(in_channels, num_points)

    def forward(self, x):
        x = self.channel_attention(x)
        # x = x * self.spatial_attention(x)
        return x




class MeshConv_CBAM(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """
    def __init__(self, in_channels, out_channels, num_edges, k=5, bias=True):
        super(MeshConv_CBAM, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)
        self.mesh_cbam = CBAMModule(in_channels, reduction_ratio=16, num_points=num_edges)
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        # print("x.shape:", x.shape)
        x = self.mesh_cbam(x)
        x = x.squeeze(-1)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        # build 'neighborhood image' and apply convolution
        G = self.create_GeMM(x, G)
        x = self.conv(G)
        return x

    def flatten_gemm_inds(self, Gi):
        (b, ne, nn) = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1 #shift

        # first flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # apply the symmetric functions for an equivariant conv
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        # print("m.gemm_edges:", m.gemm_edges)
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm


class MeshConv(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        # build 'neighborhood image' and apply convolution
        G = self.create_GeMM(x, G)
        # print("G.shape:", G.shape)
        x = self.conv(G)
        return x

    def flatten_gemm_inds(self, Gi):
        (b, ne, nn) = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1 #shift

        # first flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # apply the symmetric functions for an equivariant conv
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        # print("m.gemm_edges:", m.gemm_edges)
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm

    
    

class MeshConv_high(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv_high, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)
        # print("输入高分辨率卷积的特征- x.shape:", x.shape)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)

        # print("生成的G的尺寸- G.shape:", G.shape)
        # build 'neighborhood image' and apply convolution
        # 根据邻域索引矩阵 G 创建适合卷积操作的特征矩阵 
        G = self.create_GeMM(x, G)

        # print("用于卷积的G的尺寸-G.shape:", G.shape)
        x = self.conv(G)
        return x

    def flatten_gemm_inds(self, Gi):
        (b, ne, nn) = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1 #shift

        # first flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # # apply the symmetric functions for an equivariant conv
        # x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        # x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        # x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        # x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        # f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        
        

        
        # Apply symmetric functions for the first 5 dimensions
        f_symm = f[:, :-6, :, :]
        # print("f_symm.shape:", f_symm.shape)
        x_1 = f_symm[:, :, :, 1] + f_symm[:, :, :, 3]
        x_2 = f_symm[:, :, :, 2] + f_symm[:, :, :, 4]
        x_3 = torch.abs(f_symm[:, :, :, 1] - f_symm[:, :, :, 3])
        x_4 = torch.abs(f_symm[:, :, :, 2] - f_symm[:, :, :, 4])
        f_symm = torch.stack([f_symm[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)

        # Concatenate the symmetric processed features with the remaining 3 dimensions
        f_remaining = f[:, -6:, :, :]
        # print("f_remaining.shape:", f_remaining.shape)
        f = torch.cat([f_symm, f_remaining], dim=1)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        # print("m.gemm_edges:", m.gemm_edges)
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        # 将边索引列与 gemm_edges 列拼接，形成一个大小为 #edges x 5 的张量，每行包含边索引及其四个邻居 
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # 使用 F.pad 将张量填充到目标大小 xsz x 5。填充值为0 
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm



class SimilarityWeightGenerator(nn.Module):
    def __init__(self):
        super(SimilarityWeightGenerator, self).__init__()
        # 定义卷积层，输入通道为32，输出通道为1
        self.conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))
        self.recalculate_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))

    def forward(self, original_features, sampled_features):
        # 重复原始特征张量以匹配采样特征张量的形状
        repeated_features = original_features.unsqueeze(-1).repeat(1, 1, 1, 10)
        # 在通道维度上拼接重复的原始特征张量和采样特征张量
        concatenated_features = torch.cat((repeated_features, sampled_features), dim=1)
        # 通过卷积层处理拼接后的张量，并应用Sigmoid函数生成相似性权重
        similarity_weights = torch.sigmoid(self.conv(concatenated_features))
        # 使用torch.topk选取最高的4个权重及其索引
        topk_values, topk_indices = torch.topk(similarity_weights, k=4, dim=-1)
        # 抽取对应的邻近点特征
        topk_features = torch.gather(sampled_features, -1, topk_indices.expand(-1, 16, -1, -1))
        # 重复原始特征以匹配邻近点特征的形状
        original_expanded_1 = original_features.unsqueeze(-1).expand(-1, -1, -1, 4)
        # print("original_expanded_1.shape:", original_expanded_1.shape)
        # 在通道维度上拼接原始特征与邻近点特征
        concatenated_features = torch.cat((original_expanded_1, topk_features), dim=1)  # shape: [1, feature_dim*2, num_points, num_nearest_points]
        # 通过卷积层重新计算权重
        recalculated_weights = torch.sigmoid(self.recalculate_conv(concatenated_features))  # shape: [1, 1, num_points, num_nearest_points]
        # 使用新权重对邻近点特征进行加权
        weighted_nearest_features = topk_features * recalculated_weights
        # 对加权的邻近点特征求和以融合信息
        summed_features = weighted_nearest_features.sum(dim=-1)  # shape: [1, feature_dim, num_points]
        # 将融合后的特征与原始特征相加
        enhanced_features = original_features + summed_features
        # print("enhanced_features.shape:", enhanced_features.shape)
        return enhanced_features

class MeshTrans_all_atten(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """
    def __init__(self, in_channels, out_channels, k=10, bias=True):
        super(MeshTrans_all_atten, self).__init__()
        self.final_atten = SimilarityWeightGenerator()

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        # print("自适应卷积中的输入 x.shape:", x.shape)
        x = x.squeeze(-1)

        # print("此时x的特征：", x.shape)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        # build 'neighborhood image' and apply convolution
        # print("此时G为：", G.shape)
        G = self.create_GeMM(x, G)

        # print("挑选后的G：", G.shape)
        final_x = self.final_atten(x, G)
        # print("注意力之后的特征：", final_x.shape)
        # x = self.conv(G)
        return final_x

    def flatten_gemm_inds(self, Gi):
        (b, ne, nn) = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)

        Gi = Gi + 1 #shift

        # print("x的特征为：", x.shape, Gi.shape)
        # first flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)
        # print("f.shape:", f.shape)

        # print("交换坐标后 f.shape:", f.shape)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        # print("m.gemm_edges:", m.gemm_edges)
        padded_gemm = torch.tensor(m.gemm_all, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        # print("---padded_gemm.shape:", padded_gemm.shape)
        # padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # print("+++padded_gemm.shape:", padded_gemm.shape)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm
