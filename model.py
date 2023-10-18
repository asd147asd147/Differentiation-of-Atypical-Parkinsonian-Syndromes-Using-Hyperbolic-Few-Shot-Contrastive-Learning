import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, z=256):
        super(SiameseNetwork,self).__init__()
        self.z = z

        self.conv = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 128, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(128),

            nn.Conv3d(128, 256, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(256),
        )

        self.fc = nn.Sequential(
            nn.Linear(256*9*6*4, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, self.z),
            # nn.ReLU(inplace=True),

            # nn.Linear(512, 128),
        )

    def activation_map(self, x):
        output = self.conv(x)
        return output

    def forward_once(self, x):
        output = self.conv(x)
        # print(output.shape)
        output = output.view(output.size()[0], -1)
        # print(output.shape)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2