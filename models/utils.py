#%% Imports

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from torchmetrics.functional import structural_similarity_index_measure

#%% Class for early stopping and saving the best performing model

class EarlyStopper:

    def __init__(self, patience=1, min_delta=0, maximize=False):
        
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        
        self.counter = 0
        self.min = np.inf

    def early_stop(self, metric, checkpoint=None, save_path=None):

        if self.maximize:
            metric = metric * (-1)
        
        if metric < self.min:
            self.min = metric
            self.counter = 0

            if checkpoint is not None and save_path is not None:
                torch.save(checkpoint, save_path)
        
        elif metric >= (self.min + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False

#%% Implementation of a weighted L1-SSIM loss combination as a recontruction objective

class MixedAELoss(nn.Module):

    def __init__(self, ssim_weight):

        super().__init__()
        self.l1_weight = 1 - ssim_weight
        self.ssim_weight = ssim_weight

    def forward(self, x_hat, x):

        l1 = l1_loss(x_hat, x)
        ssim = structural_similarity_index_measure(x_hat, x)

        return self.l1_weight * l1 + self.ssim_weight * (1 - ssim)