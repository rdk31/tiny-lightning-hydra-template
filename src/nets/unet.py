import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel


class UnetDDPM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: list[int],
        layers_per_block: int,
        downblock: str,
        upblock: str,
        add_attention: bool,
        attention_head_dim: int,
        low_condition: bool,
        timestep_condition: bool,
        global_skip_connection: bool,
        num_class_embeds: int | None = None,
    ):
        super().__init__()
        self.low_condition = low_condition
        self.timestep_condition = timestep_condition
        self.global_skip_connection = global_skip_connection
        self.divide_factor = 2 ** len(channels)

        self.backbone = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=channels,
            layers_per_block=layers_per_block,
            down_block_types=tuple(downblock for _ in range(len(channels))),
            up_block_types=tuple(upblock for _ in range(len(channels))),
            add_attention=add_attention,
            attention_head_dim=attention_head_dim,
            num_class_embeds=num_class_embeds,
        )

    def padding(self, x):
        _, _, W, H = x.shape
        desired_width = (
            (W + self.divide_factor - 1) // self.divide_factor
        ) * self.divide_factor
        desired_height = (
            (H + self.divide_factor - 1) // self.divide_factor
        ) * self.divide_factor

        # Calculate the padding needed
        padding_w = desired_width - W
        padding_h = desired_height - H

        return F.pad(x, (0, padding_h, 0, padding_w), mode="constant", value=0), W, H

    def remove_padding(self, x, W, H):
        return x[:, :, :W, :H]

    def forward(self, x_t, t, x_low=None, class_labels=None):
        x_in = torch.cat([x_t, x_low], dim=1) if self.low_condition else x_t

        # add padding to fit nearest value divisible by self.divide_factor
        x_in, W, H = self.padding(x_in)

        model_output = self.backbone(
            x_in,
            timestep=t if self.timestep_condition else 0,
            class_labels=class_labels
            if class_labels is not None
            else None,  # TODO: check num_class_embeds
        ).sample

        model_output = self.remove_padding(model_output, W, H)

        if self.global_skip_connection:
            model_output[:, :3] = model_output[:, :3] + x_t

        return model_output  # pred_x_0
