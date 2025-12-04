import typer
from pathlib import Path
from .._cli import OPTION_PROMPT_KWARGS, cli

import einops
import torch
import torch.nn.functional as F
import numpy as np
from miss_alignment.models import MissAlignment
from miss_alignment.data import EMDBDataset


class AttentionRollout3D:
    def __init__(self, model, layers_to_track):
        self.model = model
        self.layers = layers_to_track
        self.gradients = {}
        self.activations = {}

        # Register hooks for all layers
        for name, layer in self.layers.items():
            layer.register_forward_hook(self._make_save_activation(name))
            layer.register_full_backward_hook(self._make_save_gradient(name))

    def _make_save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook

    def _make_save_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()

        return hook

    def generate_attention_map(self, input_tensor):
        # reset gradients
        self.model.zero_grad()

        # For single scalar output
        output = self.model(input_tensor)
        output.backward()

        # Calculate attention maps for each layer
        attention_maps = {}
        for name in self.layers:
            weights = torch.mean(self.gradients[name], dim=(2, 3, 4), keepdim=True)
            attention = torch.sum(weights * self.activations[name], dim=1)
            attention = F.relu(attention)
            attention = attention / (torch.max(attention) + 1e-10)

            # Upsample to input resolution
            attention = F.interpolate(
                attention.unsqueeze(0),
                size=input_tensor.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

            attention_maps[name] = attention

        # Combine attention maps (you can use different strategies here)
        combined_attention = torch.zeros_like(
            attention_maps[list(attention_maps.keys())[0]]
        )
        for name in attention_maps:
            combined_attention += attention_maps[name]

        combined_attention = combined_attention / len(attention_maps)
        return combined_attention.squeeze().detach().cpu().numpy(), output


@cli.command(name="visualize_attention", no_args_is_help=True)
def visualize_attention(
    model_checkpoint: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    test_data: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    output_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
) -> None:
    """Visualize the attention span of MissAlignment."""
    model = MissAlignment.load_from_checkpoint(model_checkpoint, map_location="cpu")
    model.eval()

    layers_track = {}
    layers_track["conv1"] = model.net.conv[0]
    layers_track["conv2"] = model.net.conv[3]
    layers_track["conv3"] = model.net.conv[6]
    layers_track["conv4"] = model.net.conv[9]

    rollout = AttentionRollout3D(model, layers_track)

    data = EMDBDataset(test_data, train=False)

    for i, n in enumerate(data.mrc_files):
        img1, img2, s = data[i]
        img1 = einops.rearrange(img1, "c d h w -> 1 c d h w")
        img2 = einops.rearrange(img2, "c d h w -> 1 c d h w")
        high, low = (img1, img2) if s < 0 else (img2, img1)
        map_name = n.stem
        print(f"model = {map_name}")
        attention1, score1 = rollout.generate_attention_map(high)
        attention2, score2 = rollout.generate_attention_map(low)
        print(f"High quality map has a score of {score1.detach()}")
        print(f"Low quality map has a score of {score2.detach()}")

        np.save(
            output_directory.joinpath(f"{map_name}_high_at.npy"),
            attention1,
        )
        np.save(
            output_directory.joinpath(f"{map_name}_high.npy"),
            high.squeeze().squeeze().detach().cpu().numpy(),
        )
        np.save(
            output_directory.joinpath(f"{map_name}_low_at.npy"),
            attention2,
        )
        np.save(
            output_directory.joinpath(f"{map_name}_low.npy"),
            low.squeeze().squeeze().detach().cpu().numpy(),
        )

    return None
