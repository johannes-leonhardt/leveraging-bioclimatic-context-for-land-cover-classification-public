#%% Imports

import torch.nn as nn
from models.cbn import ConditionalBatchNorm2d

#%% Reverse ResNet18 decoder model

class ResNet18Decoder(nn.Module):

    n_blocks = [2,2,2,2]
    base_channels = 64

    def __init__(self, n_channels):
        
        super().__init__()
        
        self.unpool = nn.ConvTranspose2d(self.base_channels*8, self.base_channels*8, kernel_size=4)
        
        self.layer_1 = ReverseResLayer(in_channels=self.base_channels*8, out_channels=self.base_channels*4, n_blocks=self.n_blocks[3], upsample=True)
        self.layer_2 = ReverseResLayer(in_channels=self.base_channels*4, out_channels=self.base_channels*2, n_blocks=self.n_blocks[2], upsample=True)
        self.layer_3 = ReverseResLayer(in_channels=self.base_channels*2, out_channels=self.base_channels, n_blocks=self.n_blocks[1], upsample=True)
        self.layer_4 = ReverseResLayer(in_channels=self.base_channels, out_channels=self.base_channels, n_blocks=self.n_blocks[0], upsample=False)
        
        self.deconv = nn.ConvTranspose2d(self.base_channels, n_channels, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, aux=None):
        
        x = self.unpool(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.sigmoid(self.deconv(x))
        
        return x

class ReverseResLayer(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, upsample):

        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(ReverseResBlock(in_channels, out_channels, upsample=upsample))
        
        for _ in range(n_blocks-1):
            self.blocks.append(ReverseResBlock(out_channels, out_channels))
        
    def forward(self, x):

        for block in self.blocks:
            x = block(x)
        
        return x

class ReverseResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, upsample=False):

        super().__init__()

        stride = 2 if upsample else 1
        self.identity_layer = (in_channels != out_channels)
        
        self.conv_1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1 if upsample else 0)
        self.batchnorm_1 = nn.BatchNorm2d(out_channels)
        
        self.conv_2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channels)
        
        self.conv_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1 if upsample else 0) if self.identity_layer else None
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

#%% Reverse ResNet18 decoder model with conditional batch normalization

class ResNet18DecoderCbn(nn.Module):

    n_blocks = [2,2,2,2]
    base_channels = 64

    def __init__(self, n_channels, n_aux):
        
        super().__init__()
        
        self.unpool = nn.ConvTranspose2d(self.base_channels*8, self.base_channels*8, kernel_size=4)
        
        self.layer_1 = ReverseResLayerCbn(in_channels=self.base_channels*8, out_channels=self.base_channels*4, n_blocks=self.n_blocks[3], n_aux=n_aux, upsample=True)
        self.layer_2 = ReverseResLayerCbn(in_channels=self.base_channels*4, out_channels=self.base_channels*2, n_blocks=self.n_blocks[2], n_aux=n_aux, upsample=True)
        self.layer_3 = ReverseResLayerCbn(in_channels=self.base_channels*2, out_channels=self.base_channels, n_blocks=self.n_blocks[1], n_aux=n_aux, upsample=True)
        self.layer_4 = ReverseResLayerCbn(in_channels=self.base_channels, out_channels=self.base_channels, n_blocks=self.n_blocks[0], n_aux=n_aux, upsample=False)
        
        self.deconv = nn.ConvTranspose2d(self.base_channels, n_channels, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, aux):
        
        x = self.unpool(x)

        x = self.layer_1(x, aux)
        x = self.layer_2(x, aux)
        x = self.layer_3(x, aux)
        x = self.layer_4(x, aux)

        x = self.sigmoid(self.deconv(x))
        
        return x

class ReverseResLayerCbn(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, n_aux, upsample):

        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(ReverseResBlockCbn(in_channels, out_channels, n_aux, upsample=upsample))
        
        for _ in range(n_blocks-1):
            self.blocks.append(ReverseResBlockCbn(out_channels, out_channels, n_aux))
        
    def forward(self, x, aux):

        for block in self.blocks:
            x = block(x, aux)
        
        return x

class ReverseResBlockCbn(nn.Module):
    
    def __init__(self, in_channels, out_channels, n_aux, upsample=False):

        super().__init__()

        stride = 2 if upsample else 1
        self.identity_layer = (in_channels != out_channels)
        
        self.conv_1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1 if upsample else 0)
        self.batchnorm_1 = ConditionalBatchNorm2d(out_channels, n_aux)
        
        self.conv_2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm_2 = ConditionalBatchNorm2d(out_channels, n_aux)
        
        self.conv_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1 if upsample else 0) if self.identity_layer else None
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