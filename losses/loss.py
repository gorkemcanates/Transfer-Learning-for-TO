__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, logit, target):
        loss = self.bce(logit, target.unsqueeze(1))
        return loss
