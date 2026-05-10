import torch
import torch.nn as nn

from models.DySample import DySample
from models.FASF import FASF, FrequencyGuidedRefinement
from models.layers import BottleneckDepthwiseConv


class MLP(nn.Module):
    """Linear projection for channel alignment before multi-scale fusion."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class FrequencyAwareDecoder(nn.Module):
    """Frequency-aware multi-scale decoder used by DFECrack."""

    def __init__(self, embedding_dim, use_cross_scale=False, use_freq_refine=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_freq_refine = use_freq_refine

        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=64, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=32, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=16, embed_dim=embedding_dim)

        self.DySample_C_2 = DySample(embedding_dim, scale=2)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)

        self.fasf = FASF(
            dim=embedding_dim,
            num_scales=4,
            use_cross_scale=use_cross_scale,
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim * 4, 3, 1, 1,
                      groups=embedding_dim * 4, bias=False),
            nn.BatchNorm2d(embedding_dim * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(embedding_dim * 4, embedding_dim * 4, 1, bias=False),
            nn.BatchNorm2d(embedding_dim * 4),
            nn.SiLU(inplace=True),
        )
        self.GN_C = nn.GroupNorm(
            num_channels=embedding_dim * 4,
            num_groups=embedding_dim * 4 // 16,
        )
        self.linear_fuse = BottleneckDepthwiseConv(
            embedding_dim * 4,
            embedding_dim,
            embedding_dim // 8,
            kernel_size=1,
            padding=0,
            stride=1,
        )

        if use_freq_refine:
            self.freq_refine = FrequencyGuidedRefinement(dim=embedding_dim, window_size=16)

        self.linear_pred = BottleneckDepthwiseConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        c4, c3, c2, c1 = inputs

        b, c, h, w = c4.shape
        out_c4 = self.linear_c4(
            c4.reshape(b, c, h * w).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c4 = self.DySample_C_8(out_c4)

        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(
            c3.reshape(b, c, h * w).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c3 = self.DySample_C_4(out_c3)

        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(
            c2.reshape(b, c, h * w).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c2 = self.DySample_C_2(out_c2)

        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(
            c1.reshape(b, c, h * w).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)

        out_c = self.fasf([out_c4, out_c3, out_c2, out_c1])
        out_c = self.fusion_conv(out_c)
        out_c = self.linear_fuse(out_c)

        if self.use_freq_refine:
            out_c = self.freq_refine(out_c)

        out_c = self.dropout(out_c)
        return self.linear_pred_1(self.linear_pred(out_c))


def build_frequency_decoder(embedding_dim, decoder_type='fasf_simple'):
    if decoder_type == 'fasf_simple':
        return FrequencyAwareDecoder(
            embedding_dim,
            use_cross_scale=False,
            use_freq_refine=False,
        )
    if decoder_type == 'fasf_full':
        return FrequencyAwareDecoder(
            embedding_dim,
            use_cross_scale=True,
            use_freq_refine=True,
        )
    raise ValueError(f"Unknown decoder_type: {decoder_type}")
