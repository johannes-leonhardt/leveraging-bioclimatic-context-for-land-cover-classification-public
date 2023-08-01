#%% Imports

import torch.nn as nn
from models.cbn import ConditionalBatchNorm2d

#%% ResNet18 encoder model

class ResNet18Encoder(nn.Module):

    n_blocks = [2,2,2,2]
    base_channels = 64

    def __init__(self, n_channels, in_conv_kernel_size=2):
        
        super().__init__()
        
        self.conv = nn.Conv2d(n_channels, self.base_channels, kernel_size=in_conv_kernel_size, stride=in_conv_kernel_size)
        self.batchnorm = nn.BatchNorm2d(self.base_channels)
        self.relu = nn.ReLU()
        
        self.layer_1 = ResLayer(in_channels=self.base_channels, out_channels=self.base_channels, n_blocks=self.n_blocks[0], downsample=False)
        self.layer_2 = ResLayer(in_channels=self.base_channels, out_channels=self.base_channels*2, n_blocks=self.n_blocks[1], downsample=True)
        self.layer_3 = ResLayer(in_channels=self.base_channels*2, out_channels=self.base_channels*4, n_blocks=self.n_blocks[2], downsample=True)
        self.layer_4 = ResLayer(in_channels=self.base_channels*4, out_channels=self.base_channels*8, n_blocks=self.n_blocks[3], downsample=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, aux=None):

        x = self.relu(self.batchnorm(self.conv(x)))
        
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        
        x = self.pool(x)
        
        return x

class ResLayer(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, downsample):

        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(ResBlock(in_channels, out_channels, downsample=downsample))
        
        for _ in range(n_blocks-1):
            self.blocks.append(ResBlock(out_channels, out_channels))
        
    def forward(self, x):

        for block in self.blocks:
            x = block(x)
        
        return x

class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample=False):

        super().__init__()

        stride = 2 if downsample else 1
        self.identity_layer = (in_channels != out_channels)
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_channels)
        
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channels)
        
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) if self.identity_layer else None
        self.batchnorm_i = nn.BatchNorm2d(out_channels) if self.identity_layer else None

        self.relu = nn.ReLU()
        
    def forward(self, x):

        identity = x.clone()

        if self.identity_layer:
            identity = self.batchnorm_i(self.conv_i(identity))
        
        x = self.relu(self.batchnorm_1(self.conv_1(x)))
        x = self.batchnorm_2(self.conv_2(x))
        
        x += identity
        x = self.relu(x)
        
        return x
    
#%% ResNet18 encoder model with added context embedding

class ResNet18EncoderEmb(nn.Module):

    base_channels = 64

    def __init__(self, n_channels, n_aux, in_conv_kernel_size=2):
        
        super().__init__()
        self.n_aux = n_aux
        
        self.encoder = ResNet18Encoder(n_channels, in_conv_kernel_size=in_conv_kernel_size)
        self.emb = nn.Linear(n_aux, self.base_channels*8)
        
    def forward(self, x, aux):

        x = self.encoder(x)
        aux = self.emb(aux)

        out = x + aux.view(-1, self.base_channels*8, 1, 1)
        
        return out
    
#%% ResNet18 encoder model with conditional batch normalization

class ResNet18EncoderCbn(nn.Module):

    n_blocks = [2,2,2,2]
    base_channels = 64

    def __init__(self, n_channels, n_aux, in_conv_kernel_size=2):
        
        super().__init__()
        
        self.conv = nn.Conv2d(n_channels, self.base_channels, kernel_size=in_conv_kernel_size, stride=in_conv_kernel_size)
        self.batchnorm = ConditionalBatchNorm2d(self.base_channels, n_aux)
        self.relu = nn.ReLU()
        
        self.layer_1 = ResLayerCbn(in_channels=self.base_channels, out_channels=self.base_channels, n_blocks=self.n_blocks[0], n_aux=n_aux, downsample=False)
        self.layer_2 = ResLayerCbn(in_channels=self.base_channels, out_channels=self.base_channels*2, n_blocks=self.n_blocks[1], n_aux=n_aux, downsample=True)
        self.layer_3 = ResLayerCbn(in_channels=self.base_channels*2, out_channels=self.base_channels*4, n_blocks=self.n_blocks[2], n_aux=n_aux, downsample=True)
        self.layer_4 = ResLayerCbn(in_channels=self.base_channels*4, out_channels=self.base_channels*8, n_blocks=self.n_blocks[3], n_aux=n_aux, downsample=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, aux):

        x = self.relu(self.batchnorm(self.conv(x), aux))
        
        x = self.layer_1(x, aux)
        x = self.layer_2(x, aux)
        x = self.layer_3(x, aux)
        x = self.layer_4(x, aux)
        
        x = self.pool(x)
        
        return x

class ResLayerCbn(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, n_aux, downsample):

        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(ResBlockCbn(in_channels, out_channels, n_aux, downsample=downsample))
        
        for _ in range(n_blocks-1):
            self.blocks.append(ResBlockCbn(out_channels, out_channels, n_aux))
        
    def forward(self, x, aux):

        for block in self.blocks:
            x = block(x, aux)
        
        return x

class ResBlockCbn(nn.Module):
    
    def __init__(self, in_channels, out_channels, n_aux, downsample=False):

        super().__init__()

        stride = 2 if downsample else 1
        self.identity_layer = (in_channels != out_channels)
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm_1 = ConditionalBatchNorm2d(out_channels, n_aux)
        
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm_2 = ConditionalBatchNorm2d(out_channels, n_aux)
        
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) if self.identity_layer else None
        self.batchnorm_i = ConditionalBatchNorm2d(out_channels, n_aux) if self.identity_layer else None

        self.relu = nn.ReLU()
        
    def forward(self, x, aux):

        identity = x.clone()

        if self.identity_layer:
            identity = self.batchnorm_i(self.conv_i(identity), aux)
        
        x = self.relu(self.batchnorm_1(self.conv_1(x), aux))
        x = self.batchnorm_2(self.conv_2(x), aux)
        
        x += identity
        x = self.relu(x)
        
        return x