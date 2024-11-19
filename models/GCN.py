import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from typing import List
from torch_sparse import SparseTensor
import torch.nn.functional as F


class GCN_Encoder(nn.Module):
    def __init__(
        self,
        nhids:List[int],
        dropout:float=0.5,
        with_bn:bool=False,
        with_bias:bool=True
    ):
        super(GCN_Encoder,self).__init__()
        self.with_bn = with_bn
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.activate = F.leaky_relu
        if with_bn:
            self.bns = nn.ModuleList()
        for i in range(len(nhids)-1):
            self.layers.append(GCNConv(in_channels=nhids[i],out_channels=nhids[i+1],bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhids[i+1]))
    
    def forward(self, x, edge_index):
        adj = SparseTensor.from_edge_index(edge_index=edge_index).to_device(x.device)
        for i, layer in enumerate(self.layers):
            x = layer(x,adj)
            if self.with_bn:
                x = self.bns[i](x)
            x = self.activate(x)
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

