import math
import torch
from torch import nn

CONNECTIONS = {
    10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14],
    11: [12, 8], 12: [13, 11], 7: [0, 8], 0: [1, 7], 1: [2, 0],
    2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12],
    3: [2], 6: [5]
}

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channel_first=False):
        """
        多层感知机模块
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()  # 激活函数（通常无可学习参数）
        self.drop = nn.Dropout(drop)  # Dropout 无可学习参数

        if channel_first:
            # 可学习参数：Conv2d 内的权重和偏置
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        else:
            # 可学习参数：Linear 内的权重和偏置
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LearnableGraphConv(nn.Module):
    """
    可学习的图卷积层，用于语义图卷积操作。
    """
    def __init__(self, in_features, out_features, adj, bias=True):
        super(LearnableGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 可学习参数：权重矩阵 W，形状为 [2, in_features, out_features]
        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 可学习参数：权重矩阵 M，形状为 [adj.size(0), out_features]
        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        # 固定邻接矩阵（非可学习）
        self.register_buffer('adj', adj)

        # 可学习参数：增量邻接矩阵
        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            # 可学习参数：偏置项
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, J, C = input.size()
        h0 = torch.matmul(input, self.W[0])  # [B, T, J, C']
        h1 = torch.matmul(input, self.W[1])  # [B, T, J, C']

        adj = self.adj + self.adj2  # 固定邻接矩阵加上可学习部分
        adj = (adj.T + adj) / 2    # 保证对称

        E = torch.eye(adj.size(0), dtype=torch.float, device=input.device)

        if self.adj.size(0) == T:
            # Temporal mode
            h0 = (self.M * h0).view(B * J, T, self.out_features)
            h1 = (self.M * h1).view(B * J, T, self.out_features)

            adj_expanded = adj.unsqueeze(0).repeat(B * J, 1, 1)
            E_expanded = E.unsqueeze(0).repeat(B * J, 1, 1)

            output = torch.bmm(adj_expanded * E_expanded, h0) + torch.bmm(adj_expanded * (1 - E_expanded), h1)

            if self.bias is not None:
                output = output + self.bias.view(1, 1, -1)

            output = output.view(B, J, T, self.out_features)
        else:
            # Spatial mode
            h0 = (self.M * h0).view(B * T, J, self.out_features)
            h1 = (self.M * h1).view(B * T, J, self.out_features)

            adj_expanded = adj.unsqueeze(0).repeat(B * T, 1, 1)
            E_expanded = E.unsqueeze(0).repeat(B * T, 1, 1)

            output = torch.bmm(adj_expanded * E_expanded, h0) + torch.bmm(adj_expanded * (1 - E_expanded), h1)

            if self.bias is not None:
                output = output + self.bias.view(1, 1, -1)

            output = output.view(B, T, J, self.out_features)

        return output


class KPA(nn.Module):
    """
    空间图卷积模块（KPA），基于 LearnableGraphConv。
    此处将原来的点卷积替换为空间卷积：
    输入先 reshape 为 [B*T, J, C]，再转换为 [B*T, C, J]，
    使用 1D 卷积（核大小 3，padding=1）沿节点维度进行卷积，
    然后还原为原始形状 [B, T, J, C]。
    """
    def __init__(self, adj, input_dim, output_dim, p_dropout=None, mlp_hidden_dim=None):
        super(KPA, self).__init__()

        self.gconv = LearnableGraphConv(in_features=input_dim, out_features=output_dim, adj=adj)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

        # 使用 Conv1d 实现空间卷积：作用于节点维度
        self.spatial_conv = nn.Conv1d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=3,
            padding=1
        )

        # MLP 层（卷积后）
        self.mlp = MLP(in_features=output_dim, hidden_features=mlp_hidden_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的输入形状为 [B, T, J, C]
        x = self.gconv(x)          # [B, T, J, C']
        x = self.relu(x)
        x = self.layer_norm(x)
        if self.dropout is not None:
            x = self.dropout(x)

        B, T, J, C = x.size()
        # 将输入 reshape 为 [B*T, J, C]
        x_spatial = x.view(B * T, J, C)
        # 调整为 [B*T, C, J] 以适应 Conv1d（在节点维度上卷积）
        x_spatial = x_spatial.permute(0, 2, 1)
        x_spatial = self.spatial_conv(x_spatial)
        # 恢复为 [B*T, J, C]
        x_spatial = x_spatial.permute(0, 2, 1)
        # 恢复回原始形状 [B, T, J, C]
        x = x_spatial.view(B, T, J, C)

        x = self.mlp(x)
        return x


class TPA(nn.Module):
    """
    时间图卷积模块（TPA），基于 LearnableGraphConv。
    此处将原来的点卷积替换为时序卷积：
    输入先 reshape 为 [B*J, T, C]，再转换为 [B*J, C, T]，
    使用 1D 卷积（核大小 3，padding=1）沿时间维度进行卷积，
    然后还原为原始形状 [B, J, T, C]。
    """
    def __init__(self, adj_temporal, input_dim, output_dim, p_dropout=None, mlp_hidden_dim=None):
        super(TPA, self).__init__()
        self.gconv = LearnableGraphConv(in_features=input_dim, out_features=output_dim, adj=adj_temporal)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

        # 使用 Conv1d 实现时序卷积：作用于时间维度
        self.temporal_conv = nn.Conv1d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=3,
            padding=1
        )

        # MLP 层（卷积后）
        self.mlp = MLP(in_features=output_dim, hidden_features=mlp_hidden_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的输入形状为 [B, J, T, C]
        x = self.gconv(x)          # [B, J, T, C']
        x = self.relu(x)
        x = self.layer_norm(x)
        if self.dropout is not None:
            x = self.dropout(x)

        B, J, T, C = x.size()
        # 将输入 reshape 为 [B*J, T, C]
        x_temporal = x.view(B * J, T, C)
        # 调整为 [B*J, C, T] 以适应 Conv1d（在时间维度上卷积）
        x_temporal = x_temporal.permute(0, 2, 1)
        x_temporal = self.temporal_conv(x_temporal)
        # 恢复为 [B*J, T, C]
        x_temporal = x_temporal.permute(0, 2, 1)
        # 恢复回原始形状 [B, J, T, C]
        x = x_temporal.view(B, J, T, C)

        x = self.mlp(x)
        return x


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial',
                 use_temporal_similarity=True, temporal_connection_len=1, connections=None,
                 p_dropout=None):
        """
        图卷积网络模块，支持空间或时序模式
        """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode must be either 'spatial' or 'temporal'"

        self.mode = mode
        self.neighbour_num = neighbour_num
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections

        if self.mode == 'spatial':
            adj_spatial = self._init_spatial_adj()
            self.kpa = KPA(adj=adj_spatial, input_dim=dim_in, output_dim=dim_out, p_dropout=p_dropout)
            self.spatial_gconv = LearnableGraphConv(in_features=dim_out, out_features=dim_out, adj=adj_spatial, bias=True)
            self.spatial_layer_norm = nn.LayerNorm(dim_out)
            self.spatial_relu = nn.ReLU()
        elif self.mode == 'temporal':
            adj_temporal = self._init_temporal_adj(temporal_connection_len)
            self.tpa = TPA(adj_temporal=adj_temporal, input_dim=dim_in, output_dim=dim_out, p_dropout=p_dropout)
            self.temporal_gconv = LearnableGraphConv(in_features=dim_out, out_features=dim_out, adj=adj_temporal, bias=True)
            self.temporal_layer_norm = nn.LayerNorm(dim_out)
            self.temporal_relu = nn.ReLU()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS
        for i in range(self.num_nodes):
            connected_nodes = connections.get(i, [])
            for j in connected_nodes:
                adj[i, j] = 1
        return adj  # 固定邻接矩阵

    def _init_temporal_adj(self, connection_length):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(connection_length + 1):
                if i + j < self.num_nodes:
                    adj[i, i + j] = 1
        return adj  # 固定邻接矩阵

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'spatial':
            x = self.kpa(x)  # [B, T, J, C']
            x = self.spatial_relu(x)
            x = self.spatial_layer_norm(x)
        elif self.mode == 'temporal':
            x = x.transpose(1, 2)  # [B, J, T, C]
            x = self.tpa(x)          # [B, J, T, C']
            x = self.temporal_relu(x)
            x = self.temporal_layer_norm(x)
            x = x.transpose(1, 2)    # 恢复为 [B, T, J, C']
        return x
    



