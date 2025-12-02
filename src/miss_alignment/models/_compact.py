import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool3D(nn.Module):
    """
    3D BlurPool for anti-aliased downsampling.

    Applies a low-pass filter before subsampling to reduce aliasing artifacts.
    Based on "Making Convolutional Networks Shift-Invariant Again" (Zhang, 2019).

    Parameters
    ----------
    channels : int
        Number of input channels
    pad_type : str
        Padding type: 'replicate' or 'zero'
    filt_size : int
        Size of the binomial filter (1-7)
    stride : int
        Downsampling stride
    pad_off : int
        Additional padding offset
    """

    def __init__(
        self,
        channels: int,
        pad_type: str = "replicate",
        filt_size: int = 4,
        stride: int = 2,
        pad_off: int = 0,
    ):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.stride = stride
        self.channels = channels

        # Padding for 3D: (left, right, top, bottom, front, back)
        pad_size_low = int((filt_size - 1) / 2)
        pad_size_high = int((filt_size - 1) / 2 + 0.5)
        self.pad_sizes = [
            pad_size_low + pad_off,
            pad_size_high + pad_off,
        ] * 3

        # Binomial filter coefficients (Pascal's triangle)
        coeffs = {
            1: [1.0],
            2: [1.0, 1.0],
            3: [1.0, 2.0, 1.0],
            4: [1.0, 3.0, 3.0, 1.0],
            5: [1.0, 4.0, 6.0, 4.0, 1.0],
            6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
            7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        }
        a = torch.tensor(coeffs[filt_size])

        # Create 3D filter as outer product of 1D filters
        filt = a[:, None, None] * a[None, :, None] * a[None, None, :]
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None, :, :, :].repeat(channels, 1, 1, 1, 1))

        # Padding layer
        if pad_type in ["repl", "replicate"]:
            self.pad = nn.ReplicationPad3d(self.pad_sizes)
        elif pad_type == "zero":
            self.pad = nn.ConstantPad3d(self.pad_sizes, 0)
        else:
            raise ValueError(f"Pad type [{pad_type}] not supported for 3D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            if self.pad_off == 0:
                return x[:, :, :: self.stride, :: self.stride, :: self.stride]
            else:
                return self.pad(x)[:, :, :: self.stride, :: self.stride, :: self.stride]
        else:
            return F.conv3d(self.pad(x), self.filt, stride=self.stride, groups=x.shape[1])


class Compact3DConvNet(nn.Module):
    def __init__(self):
        super(Compact3DConvNet, self).__init__()

        # Feature extraction with progressive downsampling using BlurPool
        self.conv = nn.Sequential(
            # Layer 0: 64x64x64 -> 64x64x64
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.SiLU(),
            # Layer 1: 64x64x64 -> 32x32x32
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.SiLU(),
            BlurPool3D(channels=16),
            # Layer 2: 32x32x32 -> 16x16x16
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            BlurPool3D(channels=32),
            # Layer 3: 16x16x16 -> 8x8x8
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            BlurPool3D(channels=64),
            # Layer 4: 8x8x8 -> 4x4x4
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            BlurPool3D(channels=64),
            # Layer 5: 4x4x4 -> 2x2x2
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            # Global average pooling: 2x2x2 -> 1x1x1
            nn.AdaptiveAvgPool3d(1),
        )

        # Shared feature layer
        self.features = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )

        # Score head
        self.score_head = nn.Linear(16, 1, bias=False)

        # Log-precision head for uncertainty estimation
        self.log_precision_head = nn.Linear(16, 1)

    def forward(self, x):
        # Assuming input shape: (batch_size, 1, 64, 64, 64)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64)
        x = self.features(x)  # Shared features: (batch_size, 16)
        score = self.score_head(x)  # (batch_size, 1)
        log_precision = self.log_precision_head(x)  # (batch_size, 1)
        return score, log_precision


class Compact3DConvNetGELU(nn.Module):
    def __init__(self):
        super(Compact3DConvNetGELU, self).__init__()

        # Feature extraction with progressive downsampling
        self.conv = nn.Sequential(
            # Layer 1: 64x64x64 -> 32x32x32
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.GELU(),
            # Layer 2: 32x32x32 -> 16x16x16
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            # Layer 3: 16x16x16 -> 8x8x8
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            # Layer 4: 8x8x8 -> 4x4x4
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            # Global average pooling: 4x4x4 -> 1x1x1
            nn.AdaptiveAvgPool3d(1),
        )

        # Score head
        self.score_head = nn.Linear(32, 1)

        # Log-precision head for uncertainty estimation
        self.log_precision_head = nn.Linear(32, 1)

    def forward(self, x):
        # Assuming input shape: (batch_size, 1, 64, 64, 64)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 32)
        score = self.score_head(x)  # (batch_size, 1)
        log_precision = self.log_precision_head(x)  # (batch_size, 1)
        return score, log_precision


class Compact3DConvNetSpread(nn.Module):
    def __init__(self):
        super(Compact3DConvNetSpread, self).__init__()

        # Feature extraction with progressive downsampling
        self.conv = nn.Sequential(
            # Layer 1: 64x64x64 -> 32x32x32
            nn.Conv3d(1, 8, kernel_size=7, stride=2, padding=3),
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
            nn.AdaptiveAvgPool3d(1),
        )

        # Score head
        self.score_head = nn.Linear(32, 1)

        # Log-precision head for uncertainty estimation
        self.log_precision_head = nn.Linear(32, 1)

    def forward(self, x):
        # Assuming input shape: (batch_size, 1, 64, 64, 64)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 32)
        score = self.score_head(x)  # (batch_size, 1)
        log_precision = self.log_precision_head(x)  # (batch_size, 1)
        return score, log_precision


class Compact3DConvNetWide(nn.Module):
    def __init__(self):
        super(Compact3DConvNetWide, self).__init__()

        # Feature extraction with progressive downsampling
        self.conv = nn.Sequential(
            # Layer 1: 64x64x64 -> 32x32x32
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # Layer 2: 32x32x32 -> 16x16x16
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Layer 3: 16x16x16 -> 8x8x8
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # Layer 4: 8x8x8 -> 4x4x4
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # Global average pooling: 4x4x4 -> 1x1x1
            nn.AdaptiveAvgPool3d(1),
        )

        # Score head
        self.score_head = nn.Linear(64, 1)

        # Log-precision head for uncertainty estimation
        self.log_precision_head = nn.Linear(64, 1)

    def forward(self, x):
        # Assuming input shape: (batch_size, 1, 64, 64, 64)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64)
        score = self.score_head(x)  # (batch_size, 1)
        log_precision = self.log_precision_head(x)  # (batch_size, 1)
        return score, log_precision


class Compact3DConvNetDeep(nn.Module):
    def __init__(self):
        super(Compact3DConvNetDeep, self).__init__()

        self.conv = nn.Sequential(
            # Block 1: 64x64x64 -> 32x32x32
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            # Block 2: 32x32x32 -> 16x16x16
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # Block 3: 16x16x16 -> 8x8x8
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Block 4: 8x8x8 -> 4x4x4
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Global average pooling
            nn.AdaptiveAvgPool3d(1),
        )

        # Score head
        self.score_head = nn.Linear(32, 1)

        # Log-precision head for uncertainty estimation
        self.log_precision_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        score = self.score_head(x)
        log_precision = self.log_precision_head(x)
        return score, log_precision


class Bottleneck3D(nn.Module):
    """
    3D Bottleneck block with residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    bottleneck_channels : int
        Number of channels in the bottleneck (middle 3x3x3 layer)
    stride : int, default=1
        Stride for the 3x3x3 convolution (use 2 for downsampling)
    expansion : int, default=2
        Expansion factor for output channels
    """

    def __init__(self, in_channels, bottleneck_channels, stride=1, expansion=2):
        super().__init__()

        out_channels = bottleneck_channels * expansion

        self.conv1 = nn.Conv3d(
            in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(bottleneck_channels)

        self.conv2 = nn.Conv3d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(bottleneck_channels)

        self.conv3 = nn.Conv3d(
            bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.gelu = nn.GELU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.gelu(out)

        return out


class CompactResNet3D(nn.Module):
    """
    Compact 3D ResNet for scalar regression with ~45K parameters.

    Uses wider channels to match parameter count of original network
    while maintaining bottleneck depth benefits.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Score and log_precision outputs, each of shape (batch_size, 1)
    """

    def __init__(self):
        super().__init__()

        # Initial convolution: 64^3 -> 64^3, 1 -> 24 channels
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.gelu = nn.GELU()

        # Stage 1: 64^3 -> 32^3, 16 -> 16
        self.stage1 = Bottleneck3D(16, 16, stride=2, expansion=2)

        # Stage 2: 32^3 -> 16^3, 32 -> 32
        self.stage2 = Bottleneck3D(32, 16, stride=2, expansion=2)

        # Stage 3: 16^3 -> 8^3, 32 -> 64
        self.stage3 = Bottleneck3D(32, 32, stride=2, expansion=2)

        # Stage 4: 8^3 -> 4^3, 64 -> 64
        self.stage4 = Bottleneck3D(64, 32, stride=2, expansion=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Score head
        self.score_head = nn.Linear(64, 1)

        # Log-precision head for uncertainty estimation
        self.log_precision_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        score = self.score_head(x)
        log_precision = self.log_precision_head(x)
        return score, log_precision