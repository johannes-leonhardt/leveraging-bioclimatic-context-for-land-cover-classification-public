#%% Imports

import torch.nn as nn

#%% A simple one layer perceptron classification head

class Classifier(nn.Module):

    def __init__(self, n_inputs, n_outputs):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_outputs)
        )

    def forward(self, x):
        
        return self.layers(x)