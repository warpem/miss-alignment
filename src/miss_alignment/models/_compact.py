import torch.nn as nn


class Compact3DConvNet(nn.Module):
    def __init__(self):
        super(Compact3DConvNet, self).__init__()

        # Feature extraction with progressive downsampling
        self.conv = nn.Sequential(
            # Layer 1: 64x64x64 -> 32x32x32
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            # Layer 2: 32x32x32 -> 16x16x16
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            # Layer 3: 16x16x16 -> 8x8x8
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # Layer 4: 8x8x8 -> 4x4x4
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # Global average pooling: 4x4x4 -> 1x1x1
            nn.AdaptiveAvgPool3d(1)
        )

        # Final regression layer
        self.regressor = nn.Linear(32, 1)
        # TODO add other linear layer

    def forward(self, x):
        # Assuming input shape: (batch_size, 1, 64, 64, 64)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 32)
        x = self.regressor(x)  # Final output: (batch_size, 1)
        return x