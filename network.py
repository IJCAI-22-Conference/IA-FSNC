import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull

dataset = Amazon(root='./data', name='Photo')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # graph convolution
        self.conv1 = GCNConv(dataset.num_node_features, 100)
        self.conv2 = GCNConv(100, 60)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(60, 4),
        )
        # generate G
        self.gener = nn.Sequential(
            nn.Linear(60, 60),
        )

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    # extract node embeding
    def embeding(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x

    # input node embeding,   gener + classifier, train gener
    def class1gener(self, x):
        x = self.gener(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    # partA classifier
    def classify(self, x):
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    # 计算一轮 self-traing 中置信度高的的节点
    def confidence(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)

        softmax = F.softmax(x, dim=1)
        log_softmax = F.log_softmax(x, dim=1)
        comentropy = (-1) * (softmax.mul(log_softmax).sum(axis=1))
        comentropy = comentropy.reshape(len(comentropy), 1)
        pre_label = softmax.argmax(dim=1)  # 预测的类别
        pre_label = pre_label.reshape(len(pre_label), 1)

        # number, comentropy, pseudo-label
        linshi = np.arange(len(data.y))
        linshi = linshi[:, np.newaxis]
        new_x = np.hstack([linshi, comentropy.detach().cpu().numpy()])
        new_x = np.hstack([new_x, pre_label.detach().cpu().numpy()])

        return new_x




## fine-turning网络
class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # graph convolution
        self.conv1 = GCNConv(dataset.num_node_features, 100)
        self.conv2 = GCNConv(100, 60)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(60, 2),
        )
        # generate G
        self.gener = nn.Sequential(
            nn.Linear(60, 60),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    # extract node embeding
    def embeding(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x

    # input node embeding,   gener + classifier, train gener
    def class1gener(self, x):
        x = self.gener(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    # partA classifier
    def classify(self, x):
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    # 计算一轮 self-traing 中置信度高的的节点
    def confidence(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)

        softmax = F.softmax(x, dim=1)
        log_softmax = F.log_softmax(x, dim=1)
        comentropy = (-1) * (softmax.mul(log_softmax).sum(axis=1))
        comentropy = comentropy.reshape(len(comentropy), 1)
        pre_label = softmax.argmax(dim=1)  # 预测的类别
        pre_label = pre_label.reshape(len(pre_label), 1)

        # number, comentropy, pseudo-label
        linshi = np.arange(len(data.y))
        linshi = linshi[:, np.newaxis]
        new_x = np.hstack([linshi, comentropy.detach().cpu().numpy()])
        new_x = np.hstack([new_x, pre_label.detach().cpu().numpy()])

        return new_x