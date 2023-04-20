# --------------------------------------------------------
# Transfer Learning for Topology Optimization
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------


import torch
from torch import nn
import numpy as np

def plot(a):
    plt.figure()
    plt.imshow(a)

class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        preds = torch.round(yhat).squeeze(1)
        target = torch.round(y)
        acc = (torch.sum(preds == target)) / (preds.size(0) * preds.size(1) *
                                         preds.size(2))
        return acc.detach().cpu().numpy().item()

class Precision(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = torch.round(yhat).squeeze(1)
        target = torch.round(y)
        TP = torch.sum((preds == 1) * (target == 1))
        FP = torch.sum((preds == 1) * (target == 0))
        return (TP / (TP + FP + self.eps)).detach().cpu().numpy().item()


class Recall(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = torch.round(yhat).squeeze(1)
        target = torch.round(y)
        TP = torch.sum((preds == 1) * (target == 1))
        FN = torch.sum((preds == 0) * (target == 1))
        return (TP / (TP + FN + self.eps)).detach().cpu().numpy().item()
