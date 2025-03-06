import torch

from miss_alignment.models import resnet3d_18


def test_resnet():
    sample_input = torch.randn(2, 1, 128, 128, 128)

    # Create a ResNet-18 model
    model = resnet3d_18(num_input_channels=1, num_classes=1)

    # Forward pass
    output = model(sample_input)

    assert output.shape == (2, 1)
