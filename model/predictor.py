"""
-------------------------------------------------
# ProjectName：GNN_Co_Inference
# FileName: GIN.py
# Author: zhouao
# Date: 2023/10/13
-------------------------------------------------
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge, global_add_pool,global_max_pool


class GIN0(torch.nn.Module):
    def __init__(self, num_features, num_layers=3, hidden1=1024,hidden2=1024,hidden3=1024, aggr='mean', dropout=0.5,pool="sum"):
        super(GIN0, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden1),
                ReLU(),
                Linear(hidden1, hidden1),
                ReLU(),
                BN(hidden1),
            ), train_eps=False,aggr=aggr)
        self.aggr = aggr
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden1, hidden2),
                            ReLU(),
                            Linear(hidden2, hidden2),
                            ReLU(),
                            BN(hidden2),
                        ), train_eps=False,aggr=aggr))
            elif i == 1:
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden2, hidden2),
                            ReLU(),
                            Linear(hidden2, hidden2),
                            ReLU(),
                            BN(hidden2),
                        ), train_eps=False,aggr=aggr))
            else:
                sys.out()
        if "||" in pool:
            self.lin1 = Linear(hidden2*2, hidden3)
        else:
            self.lin1 = Linear(hidden2, hidden3)
        self.lin2 = Linear(hidden3, 1)

        self.pooling = pool
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum||max':
            x1 = global_add_pool(x, batch)
            x2 = global_max_pool(x, batch)
            x = torch.cat([x1, x2], dim=1)
        elif self.pooling == 'sum||mean':
            x1 = global_add_pool(x, batch)
            x2 = global_mean_pool(x, batch)
            x = torch.cat([x1, x2], dim=1)
        elif self.pooling == 'mean||max':
            x1 = global_mean_pool(x, batch)
            x2 = global_max_pool(x, batch)
            x = torch.cat([x1, x2], dim=1)
        else:
            sys.exit()
            print("wrong pool")
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)

        return x

    def __repr__(self):
        return self.__class__.__name__


