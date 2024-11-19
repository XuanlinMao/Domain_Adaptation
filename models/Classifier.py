import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class Classifier(torch.nn.Module):
    def __init__(
        self,
        nhids:List[int],
        dropout:float=0.5,
        with_bn:bool=False,
        with_bias:bool=True
    ):
        super(Classifier, self).__init__()
        self.with_bn = with_bn
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.activate = F.leaky_relu
        self.sigmoid = nn.Sigmoid()
        if with_bn:
            self.bns = nn.ModuleList([])
        for i in range(len(nhids)-2):
            self.layers.append(nn.Linear(in_features=nhids[i],out_features=nhids[i+1],bias=with_bias))
            if self.with_bn:
                self.bns.append(nn.BatchNorm1d(nhids[i+1]))
        self.layers.append(nn.Linear(in_features=nhids[-2],out_features=nhids[-1],bias=with_bias))

    def forward(self,x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.with_bn:
                x = self.bns[i](x)
            x = self.activate(x)
            if self.dropout:
                x = F.dropout(x,p=self.dropout,training=self.training)
        x = self.activate(self.layers[-1](x))
        x = self.sigmoid(x)
        return x
    
    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()