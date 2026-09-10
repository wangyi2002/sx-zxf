import math

import torch
from torch import nn

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}


class GCN(nn.Module):
    def __init__(self, dim=64, num_nodes=17, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None, ):    
        self.nodes_ = """
        :param num_nodes: Number of nodes
        :param neighbour_num: Neighbor numbers. Used in temporal GCN to create edges
        :param mode: Either 'spatial' or 'temporal'
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param connections: Spatial connections for graph edges (Optional)
        """
        super().__init__()
        assert mode in ['spatial', 'temporal'], "Mode is undefined"

        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim = dim
        self.mode = mode
        self.use_temporal_similarity = use_temporal_similarity
        self.num_nodes = num_nodes
        self.connections = connections

        if mode == 'spatial':
            self.adj=self._init_spatial_adj() 
            
            
        elif mode == 'temporal' and not self.use_temporal_similarity:
            if not self.use_temporal_similarity:
                self.adj = self._init_temporal_adj(temporal_connection_len)
            else:
                self.adj = torch.zeros((self.num_nodes, self.num_nodes))
                # self.register_buffer('adj', torch.zeros((self.num_nodes, self.num_nodes))) 


        self.W = nn.Parameter(torch.zeros(size=(2, dim, dim), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(self.num_nodes, dim), dtype=torch.float))
        self.adj2 = nn.Parameter(torch.ones(size=(self.num_nodes, self.num_nodes), dtype=torch.float))

        nn.init.constant_(self.adj2, 1e-6)

        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

        

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
    
        return adj


    def _init_temporal_adj(self, connection_length):
        """Connects each joint to itself and the same joint withing next `connection_length` frames."""
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            try:
                for j in range(connection_length + 1):
                    adj[i, i + j] = 1
            except IndexError:  # next j frame does not exist
                pass
        return adj
    
    

# #基于节点度的归一化
#     @staticmethod 
#     def normaliez_adj(adj,add_self_loops=True):
#         if add_self_loops:
#             eye=torch.eye(adj.size(-1),dtype=adj.dtype, device=adj.device)
#             adj=adj+eye

#         degree=torch.sum(adj,dim=-1)
#         degree=degree.clamp(min=1e-6) 
#         de_sqrt=torch.pow(degree,-0.5)
        
#         if adj.dim()==2:
#             degree_matrix_inv_sqrt=torch.diag(de_sqrt)
#         else:
#             degree_matrix_inv_sqrt=torch.diag_embed(de_sqrt)
#         adj_normalized = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)
        
#         return adj_normalized



    def forward(self, x):
        b, t, j, c = x.shape
        if self.mode == 'temporal':
            x = x.transpose(1, 2)# (B, T, J, C) -> (B, J, T, C)
            x = x.reshape(-1, t, c)
        else:
            x = x.reshape(-1, j, c)

        h0 = torch.matmul(x, self.W[0])
        h1 = torch.matmul(x, self.W[1])

        if self.mode == 'temporal':
                similarity = x @ x.transpose(1, 2)
                threshold = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].view(b * j, t, 1)
                adj = (similarity >= threshold).float()
        else:
            adj = self.adj
        

        adj = adj.to(x.device) + self.adj2.to(x.device)
        E=torch.eye(adj.size(1),dtype=torch.float).to(x.device)
        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)

        if self.mode == 'spatial' :
            output = output.reshape(b, t, j, self.dim)
        else:
            output = output.reshape(b, j, t, self.dim).transpose(1, 2)
        # print (self.M)
        return output
    
# if  __name__=='__main__':
#     gcn=GCN(mode='spatial')
    
   

