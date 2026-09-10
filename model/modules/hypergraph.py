import math
import torch
from torch import nn

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}

# 预定义超边分组（例如：手臂、腿等）
HYPER_EDGES = [
    [0, 1, 2, 3, 4],  # head
    [6, 8, 10],  # rarm
    [5, 7, 9],  # larm
    [12, 14, 16],  # rleg
    [11, 13, 15]  # lleg
]


class HGN(nn.Module):
    def __init__(self, dim=64, num_nodes=17, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None):
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"

        # 原有参数初始化
        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim = dim
        self.mode = mode
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections

        # 初始化邻接矩阵
        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
        elif mode == 'temporal' and not self.use_temporal_similarity:
            self.adj = self._init_temporal_adj(temporal_connection_len)
        else:
            self.adj = torch.zeros((self.num_nodes, self.num_nodes))

        # 超图参数
        self.hyper_adj = self._init_hyper_adj()  # 超图关联矩阵
        self.W_hyper = nn.Parameter(torch.zeros(size=(2, dim, dim), dtype=torch.float))
        nn.init.xavier_uniform_(self.W_hyper.data, gain=1.414)

        # 原有可学习参数
        self.W = nn.Parameter(torch.zeros(size=(2, dim, dim), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.M = nn.Parameter(torch.ones(size=(self.num_nodes, dim), dtype=torch.float))
        self.adj2 = nn.Parameter(torch.ones(size=(self.num_nodes, self.num_nodes), dtype=torch.float))
        nn.init.constant_(self.adj2, 1e-6)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

    def _init_hyper_adj(self):
        """构建超图关联矩阵H（nodes x hyper_edges）"""
        H = torch.zeros((self.num_nodes, len(HYPER_EDGES)))
        for e, nodes in enumerate(HYPER_EDGES):
            for node in nodes:
                H[node, e] = 1
        return H

    def _init_spatial_adj(self):
        # 原有空间邻接矩阵构建
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS
        for i in range(self.num_nodes):
            for j in connections.get(i, []):
                adj[i, j] = 1
        return adj

    def _init_temporal_adj(self, connection_length):
        # 原有时间邻接矩阵构建
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(connection_length + 1):
                if i + j < self.num_nodes:
                    adj[i, i + j] = 1
        return adj

    def forward(self, x):
        b, t, j, c = x.shape
        if self.mode == 'temporal':
            x = x.transpose(1, 2).reshape(-1, t, c)
        else:
            x = x.reshape(-1, j, c)

        # 原有特征变换
        h0 = torch.matmul(x, self.W[0])
        h1 = torch.matmul(x, self.W[1])

        # 超图分支
        if self.mode == 'spatial':
            H0 = torch.matmul(x, self.W_hyper[0])
            H1 = torch.matmul(x, self.W_hyper[1])
            hyper_adj = self.hyper_adj
            hyper_adj = hyper_adj.to(x.device)
            E_hyper_adj = torch.eye(hyper_adj.size(1), dtype=torch.float).to(x.device)
            Hyper_output = torch.matmul(hyper_adj*E_hyper_adj, H0)+torch.matmul(hyper_adj*E_hyper_adj, H1)


        # 动态邻接矩阵
        if self.mode == 'temporal':
            similarity = x @ x.transpose(1, 2)
            threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
            adj = (similarity >= threshold).float()
        else:
            adj = self.adj

        adj = adj.to(x.device) + self.adj2.to(x.device)
        E = torch.eye(adj.size(1), dtype=torch.float).to(x.device)

        # 原有传播
        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        # 融合超图特征

        # 调整形状
        if self.mode == 'spatial':
            output = output + Hyper_output
            output = output.reshape(b, t, j, self.dim)
        else:
            output = output.reshape(b, j, t, self.dim).transpose(1, 2)
        return output