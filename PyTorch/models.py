import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        class InputTransformNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(in_channels=1,
                              out_channels=64,
                              kernel_size=(1, 3)),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64,
                              out_channels=128,
                              kernel_size=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128,
                              out_channels=1024,
                              kernel_size=(1, 1)),
                    nn.ReLU(inplace=True)
                )
                self.linear_layers = nn.Sequential(
                    nn.Linear(in_features=1024,
                              out_features=512),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=512,
                              out_features=256),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=256,
                              out_features=3*3)
                )
                self.bias = torch.eye(3, requires_grad=True)

            def forward(self, inputs):
                '''                
                Arguments:
                    inputs {tensor} -- shape:$$(N,points,3)$$
                '''
                inputs = inputs.unsqueeze(1)
                layer_output = self.conv_layers(inputs)
                pool_output = F.max_pool2d(layer_output,
                                           kernel_size=(layer_output.shape[2], 1)).view(-1, 1024)
                layer_output = self.linear_layers(pool_output)
                return layer_output.view(-1, 3, 3) + self.bias

        class FeatureTransfromNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(in_channels=64,
                              out_channels=64,
                              kernel_size=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64,
                              out_channels=128,
                              kernel_size=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128,
                              out_channels=1024,
                              kernel_size=(1, 1)),
                    nn.ReLU(inplace=True)
                )
                self.linear_layers = nn.Sequential(
                    nn.Linear(in_features=1024,
                              out_features=512),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=512,
                              out_features=256),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=256,
                              out_features=64*64)
                )
                self.bias = torch.eye(64, requires_grad=True)

            def forward(self, inputs):
                '''                
                Arguments:
                    inputs {tensor} -- shape:$$(N,64,points,1)$$
                '''
                layer_output = self.conv_layers(inputs)
                pool_output = F.max_pool2d(layer_output,
                                           kernel_size=(layer_output.shape[2], 1)).view(-1, 1024)
                layer_output = self.linear_layers(pool_output)
                return layer_output.view(-1, 64, 64) + self.bias

        self.itrans = InputTransformNet()
        self.ftrans = FeatureTransfromNet()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=1024,
                      kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=1024)
        )
        self.linear_block = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=512),
            nn.Dropout(p=.3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512,
                      out_features=256),
            nn.Dropout(p=.3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256,
                      out_features=40)
        )

    def forward(self, inputs):
        '''
        Input shape: $$(N,points,3)$$
        '''
        itransformed = self.itrans(inputs)
        matmul = torch.bmm(inputs, itransformed).unsqueeze(1)
        layer_output = self.conv_block_1(matmul)
        ftransformed = self.ftrans(layer_output)
        layer_output = layer_output.squeeze(-1).permute(0, 2, 1)
        matmul = torch.bmm(layer_output, ftransformed).permute(
            0, 2, 1).unsqueeze(-1)
        layer_output = self.conv_block_2(matmul)
        pool_output = F.max_pool2d(layer_output,
                                   kernel_size=(layer_output.shape[2], 1)).view(-1, 1024)
        return self.linear_block(pool_output)
