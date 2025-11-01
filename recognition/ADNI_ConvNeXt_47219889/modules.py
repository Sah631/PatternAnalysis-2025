"""
Core model architecture definitions.

This module implements the ConvNeXt architecture and its key components:
- DropPath: stochastic depth regularisation
- LayerNormChannel: channel-wise layer normalisation for NCHW tensors
- ConvNeXtBlock: main building block with depthwise conv, MLP, and residuals
- DownsampleLayer: feature map resolution reduction
- ConvNeXt: complete model adapted for single-channel MRI classification

All components are written in PyTorch and designed for clarity, modularity, and compatibility
with medical imaging data.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    Randomly zeroes a full residual branch per sample with probability drop_prob
    and rescales to preserve expected magnitude.
    
    This is only done during training.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class LayerNormChannel(nn.Module):
    """Applies LayerNorm across channels by permuting NCHW to NHWC, normalising, then permuting back."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # Permute to NHWC for channel-wise normalisation, then back to NCHW
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x
    

class ConvNeXtBlock(nn.Module):
    """
    Implements a ConvNeXt block with depthwise convolution, channel-wise LayerNorm,
    pointwise MLP, optional layer scaling, and stochastic depth for regularisation.
    """
    def __init__(
            self,
            dim,
            layer_scale_init=1e-6,
            drop_path=0.0):
        super().__init__()

        self.dw_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding='same', groups=dim)

        self.ln = LayerNormChannel(dim, eps=1e-6)

        self.pw_conv1 = nn.Conv2d(in_channels=dim, out_channels=4*dim, kernel_size=1)
        self.activation = nn.GELU()
        self.pw_conv2 = nn.Conv2d(in_channels=4*dim, out_channels=dim, kernel_size=1)

        if layer_scale_init > 0:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim, 1, 1)))
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        res = x # Save input for residual connection

        # Depthwise convolution + LayerNorm
        x = self.dw_conv(x)
        x = self.ln(x)

        # Pointwise MLP (1x1 conv -> GELU -> 1x1 conv)
        x = self.pw_conv1(x)
        x = self.activation(x)
        x = self.pw_conv2(x)

        # Optional layer scaling to stabilise training
        if self.gamma is not None:
            x = self.gamma * x
        
        # Apply stochastic depth and add residual
        x = self.drop_path(x)
        x = x + res

        return x


class DownsampleLayer(nn.Module):
    """Applies Layer normalisation followed by a 2x2 convolution with stride 2 to halve the spatial size."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNormChannel(in_dim, eps=1e-6)
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        # Normalise then downsample feature maps
        x = self.norm(x)
        x = self.reduction(x)
        return x
    

class ConvNeXt(nn.Module):
    """
    ConvNeXt architecture adapted for single-channel MRI classification.
    Comprises four stages of ConvNeXt blocks with downsampling, global pooling,
    and a linear head for final class prediction.
    """
    def __init__(
        self,
        in_chans=1,
        num_classes=2,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init=1e-6,
    ):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4

        # Converts image to non-overlapping patches and projects to dims[0].
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNormChannel(dims[0], eps=1e-6),
        )


        # Drop prob is linearly increased with block index
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        dp_idx = 0

        stages = []
        cur_dim = dims[0]

        for stage_idx in range(4):
            blocks = []
            for _ in range(depths[stage_idx]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=cur_dim,
                        layer_scale_init=layer_scale_init,
                        drop_path=dp_rates[dp_idx],
                    )
                )
                dp_idx += 1
            stage = nn.Sequential(*blocks)
            stages.append(stage)

            # Downsample between stages
            if stage_idx < 3:
                next_dim = dims[stage_idx + 1]
                stages.append(DownsampleLayer(cur_dim, next_dim))
                cur_dim = next_dim

        self.features = nn.Sequential(*stages)

        self.norm = nn.LayerNorm(cur_dim, eps=1e-6)
        self.head = nn.Linear(cur_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialises model's parameters to reasonable values before training.
        Helps prevent exploding/vanishing gradients and improves stability while training.
        """
        if isinstance(m, nn.Conv2d):
            # Kaiming init for convs
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        x = self.norm(x)
        x = self.head(x)

        return x
    