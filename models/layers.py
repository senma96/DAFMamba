import torch.nn as nn


class BottleneckDepthwiseConv(nn.Module):
    """Lightweight bottleneck depthwise convolution used across DFECrack."""

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size,
            stride,
            padding,
            groups=mid_channels,
            bias=False,
        )
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class LocalContextBlock(nn.Module):
    """Local morphology-aware context block for crack feature refinement."""

    def __init__(self, in_channels, norm_type='GN'):
        super().__init__()
        self.block1 = nn.Sequential(
            BottleneckDepthwiseConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            self._norm(norm_type, in_channels, in_channels // 16),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            BottleneckDepthwiseConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            self._norm(norm_type, in_channels, in_channels // 16),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            BottleneckDepthwiseConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            self._norm(norm_type, in_channels, in_channels // 16),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            BottleneckDepthwiseConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            self._norm(norm_type, in_channels, 16),
            nn.ReLU(),
        )

    @staticmethod
    def _norm(norm_type, channels, num_groups):
        if norm_type == 'GN':
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        return nn.InstanceNorm3d(channels)

    def forward(self, x):
        residual = x
        x1 = self.block2(self.block1(x))
        x2 = self.block3(x)
        x = self.block4(x1 * x2)
        return x + residual
