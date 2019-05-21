"""
author: Nicolas
"""

import torch.nn as nn
import logging


class SimplisticFCN(nn.Module):
    """
    Simple convolutional auto-encoder neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    """
    def __init__(self, input_channels=3, output_channels=3, num_filter=24, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        output_channels : int
            Number of classes
        """

        super(SimplisticFCN, self).__init__()

        self.in_dim = input_channels
        self.out_dim = output_channels

        self.conv1_1 = nn.Conv2d(3, 32, (3,3), padding=(1,1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv1_3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.relu1_4 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv1_5 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu1_5 = nn.ReLU(inplace=True)
        self.conv1_6 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.relu1_6 = nn.ReLU(inplace=True)

        self.score_fr = nn.Conv2d(128, self.out_dim, (1,1))

        self.upscore4 = nn.ConvTranspose2d(
            self.out_dim, self.out_dim, (4,4), stride=(4,4), bias=False)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu1_3(self.conv1_3(h))
        h = self.relu1_4(self.conv1_4(h))
        h = self.pool2(h)

        h = self.relu1_5(self.conv1_5(h))
        h = self.relu1_6(self.conv1_6(h))

        h = self.score_fr(h)
        h = self.softmax(self.upscore4(h))

        return h
