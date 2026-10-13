import torch
import torch.nn as nn

import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(SegNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
         #   nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        for layer in self.encoder:
            x = layer(x)
            #print(f"Encoder Output Size: {x.size()}")  # Debug encoder output sizes

        # Decoder
        for layer in self.decoder:
            x = layer(x)
        #    print(f"Decoder Output Size: {x.size()}")  # Debug decoder output sizes

        return x


# Initialize SegNet
segnet = SegNet(in_channels=4, out_channels=1).to("cuda" if torch.cuda.is_available() else "cpu")
