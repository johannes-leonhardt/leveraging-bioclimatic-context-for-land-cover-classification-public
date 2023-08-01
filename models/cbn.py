#%% Imports

import torch.nn as nn

#%% Implementation of conditional batch normalization

class ConditionalBatchNorm2d(nn.Module):
  
    def __init__(self, n_features, n_aux):

        super().__init__()
        
        self.n_features = n_features
        self.bn = nn.BatchNorm2d(n_features, affine=False)
        self.linear = nn.Linear(n_aux, n_features * 2)

    def forward(self, x, aux):

        out = self.bn(x)
        gamma, beta = self.linear(aux).chunk(2, 1)
        out = gamma.view(-1, self.n_features, 1, 1) * out + beta.view(-1, self.n_features, 1, 1)
        
        return out