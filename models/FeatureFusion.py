import torch
import torch.nn as nn


class AdaptiveChannelSpatialFusion(nn.Module):
    """
    Adaptive channel-spatial fusion for combining local crack features and
    global state-space features.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = channels // reduction
        self.proj = nn.Conv2d(channels, mid, 1, bias=False)
        self.gate_gen = nn.Sequential(
            nn.Conv2d(mid + 2, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, base_feat, guidance_feat):
        base_proj = self.proj(base_feat)
        guidance_proj = self.proj(guidance_feat)
        interaction = base_proj * guidance_proj

        spatial_max = torch.max(guidance_feat, dim=1, keepdim=True)[0]
        spatial_avg = torch.mean(guidance_feat, dim=1, keepdim=True)
        combined = torch.cat([interaction, spatial_max, spatial_avg], dim=1)
        gate = self.gate_gen(combined)

        return gate * guidance_feat + (1 - gate) * base_feat


def build_fusion_module(fusion_type, channels):
    if fusion_type == 'acsf':
        return AdaptiveChannelSpatialFusion(channels)
    raise ValueError(f"Unknown fusion_type: {fusion_type}. The released model uses 'acsf'.")
