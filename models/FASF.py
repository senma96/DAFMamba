"""
FASF: Frequency-Aware Skip Fusion
频率感知跨尺度融合模块

核心思想：
1. SAVSS的多尺度特征是"人造的"（通过上采样生成）
2. 深层特征(c4)包含结构信息 → 提取低频
3. 浅层特征(c1)包含细节信息 → 提取高频  
4. 通过频率域融合让多尺度特征更协调

Author: Hui Liu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecompose(nn.Module):
    """
    频率分解模块
    
    将特征分解为低频和高频两部分
    """
    def __init__(self, dim, ratio=0.25):
        """
        Args:
            dim: 特征通道数
            ratio: 低频区域占比 (0.25 = 中心25%区域为低频)
        """
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            low_freq: 低频成分 [B, C, H, W]
            high_freq: 高频成分 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # FFT变换
        freq = torch.fft.rfft2(x, norm='ortho')
        
        # 创建低频掩码（中心区域）
        h_center = int(H * self.ratio)
        w_center = int((W // 2 + 1) * self.ratio)  # rfft2的宽度是W//2+1
        
        # 低频掩码
        low_mask = torch.zeros(H, W // 2 + 1, device=x.device, dtype=torch.float32)
        low_mask[:h_center, :w_center] = 1.0
        low_mask[-h_center:, :w_center] = 1.0  # fftshift后的对称部分
        
        # 分离低频和高频
        low_freq_fft = freq * low_mask.unsqueeze(0).unsqueeze(0)
        high_freq_fft = freq * (1 - low_mask.unsqueeze(0).unsqueeze(0))
        
        # IFFT变换回空间域
        low_freq = torch.fft.irfft2(low_freq_fft, s=(H, W), norm='ortho')
        high_freq = torch.fft.irfft2(high_freq_fft, s=(H, W), norm='ortho')
        
        return low_freq, high_freq


class AdaptiveFrequencyFusion(nn.Module):
    """
    自适应频率融合模块
    
    根据特征层级自适应地融合低频和高频成分
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 可学习的融合权重
        self.low_weight = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        self.high_weight = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        
        # 权重生成网络（根据特征内容动态调整）
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 2, 1),  # 输出2个权重：low和high
            nn.Sigmoid()
        )
        
    def forward(self, low_freq, high_freq, x_original):
        """
        Args:
            low_freq: 低频成分 [B, C, H, W]
            high_freq: 高频成分 [B, C, H, W]
            x_original: 原始特征，用于生成动态权重
        Returns:
            fused: 融合后的特征 [B, C, H, W]
        """
        # 动态权重
        weights = self.weight_net(x_original)  # [B, 2, 1, 1]
        dynamic_low_w = weights[:, 0:1, :, :]
        dynamic_high_w = weights[:, 1:2, :, :]
        
        # 融合：静态权重 + 动态权重
        fused = (self.low_weight * dynamic_low_w) * low_freq + \
                (self.high_weight * dynamic_high_w) * high_freq
        
        return fused


class CrossScaleFrequencyAlign(nn.Module):
    """
    跨尺度频率对齐模块
    
    核心创新：让深层特征的低频和浅层特征的高频协调融合
    """
    def __init__(self, dim, num_scales=4):
        """
        Args:
            dim: 统一后的特征维度（embedding_dim）
            num_scales: 尺度数量
        """
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # 频率分解器
        self.freq_decompose = FrequencyDecompose(dim, ratio=0.25)
        
        # 每个尺度的频率调制权重
        # 深层(c4)：强调低频; 浅层(c1)：强调高频
        self.scale_low_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, dim, 1, 1) * (0.8 - 0.2 * i))  # c4:0.8, c3:0.6, c2:0.4, c1:0.2
            for i in range(num_scales)
        ])
        self.scale_high_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, dim, 1, 1) * (0.2 + 0.2 * i))  # c4:0.2, c3:0.4, c2:0.6, c1:0.8
            for i in range(num_scales)
        ])
        
        # 跨尺度注意力：让不同尺度的频率成分相互增强
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(dim // 8, dim),
            nn.SiLU()
        )
        
    def forward(self, features):
        """
        Args:
            features: list of [B, C, H, W], 长度为num_scales
                     顺序：[c4, c3, c2, c1] (深层到浅层)
        Returns:
            aligned_features: list of [B, C, H, W]
        """
        B, C, H, W = features[0].shape
        
        # 1. 对每个尺度进行频率分解和加权
        enhanced_features = []
        all_low_freqs = []
        all_high_freqs = []
        
        for i, feat in enumerate(features):
            low_freq, high_freq = self.freq_decompose(feat)
            all_low_freqs.append(low_freq)
            all_high_freqs.append(high_freq)
            
            # 按尺度加权
            weighted = self.scale_low_weights[i] * low_freq + \
                      self.scale_high_weights[i] * high_freq
            enhanced_features.append(weighted)
        
        # 2. 跨尺度频率交互
        # 将深层的低频和浅层的高频进行交互增强
        # 简化实现：让c4的低频增强c1的结构，c1的高频细化c4的边缘
        
        # 深层低频 → 增强所有层的结构一致性
        low_freq_global = all_low_freqs[0]  # c4的低频作为全局结构
        for i in range(1, self.num_scales):
            # 将深层低频与浅层特征对齐
            enhanced_features[i] = enhanced_features[i] + 0.1 * low_freq_global
        
        # 浅层高频 → 增强所有层的边缘细节
        high_freq_detail = all_high_freqs[-1]  # c1的高频作为细节来源
        for i in range(self.num_scales - 1):
            # 将浅层高频细节传递给深层
            enhanced_features[i] = enhanced_features[i] + 0.1 * high_freq_detail
        
        # 3. 输出投影
        aligned_features = [self.out_proj(feat) + feat for feat in enhanced_features]
        
        return aligned_features


class FASF(nn.Module):
    """
    FASF: Frequency-Aware Skip Fusion
    频率感知跨尺度融合模块
    
    用于替代普通多尺度拼接，实现频率域的特征对齐和融合
    
    创新点：
    1. 识别SAVSS多尺度特征的"人造性"问题
    2. 通过频率域分解，让深层保结构、浅层保细节
    3. 跨尺度频率交互，提高特征一致性
    """
    def __init__(self, dim, num_scales=4, use_cross_scale=True):
        """
        Args:
            dim: 统一后的特征维度（embedding_dim）
            num_scales: 尺度数量（默认4）
            use_cross_scale: 是否使用跨尺度频率交互
        """
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.use_cross_scale = use_cross_scale
        
        if use_cross_scale:
            # 完整的跨尺度频率对齐
            self.cross_scale_align = CrossScaleFrequencyAlign(dim, num_scales)
        else:
            # 简化版：每个尺度独立的频率增强
            self.freq_decompose = FrequencyDecompose(dim, ratio=0.25)
            self.fusion_modules = nn.ModuleList([
                AdaptiveFrequencyFusion(dim) for _ in range(num_scales)
            ])
        
        # 融合后的特征增强
        self.fuse_enhance = nn.Sequential(
            nn.Conv2d(dim * num_scales, dim * num_scales, 3, padding=1, groups=num_scales),
            nn.GroupNorm(num_scales, dim * num_scales),
            nn.SiLU(),
            nn.Conv2d(dim * num_scales, dim * num_scales, 1),
        )
        
    def forward(self, features):
        """
        Args:
            features: list of [B, C, H, W], 长度为num_scales
                     顺序：[c4, c3, c2, c1] (深层到浅层，都已上采样到相同分辨率)
        Returns:
            fused: [B, C*num_scales, H, W] 融合后的特征
        """
        if self.use_cross_scale:
            # 跨尺度频率对齐
            aligned = self.cross_scale_align(features)
        else:
            # 独立频率增强
            aligned = []
            for i, feat in enumerate(features):
                low, high = self.freq_decompose(feat)
                enhanced = self.fusion_modules[i](low, high, feat)
                aligned.append(enhanced + feat)  # 残差
        
        # 拼接
        fused = torch.cat(aligned, dim=1)
        
        # 融合增强
        fused = self.fuse_enhance(fused) + fused
        
        return fused


class FrequencyGuidedRefinement(nn.Module):
    """
    频率引导细化模块
    
    在最终输出前，利用频率信息细化边缘
    """
    def __init__(self, dim, window_size=16):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        # 高频提取
        self.high_freq_extract = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(dim // 8, dim),
            nn.SiLU()
        )
        
        # 边缘增强
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(dim // 8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1)
        )
        
        # 可学习的高频增强系数
        self.high_freq_gain = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.3)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            refined: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 局部FFT提取高频
        # 使用滑动窗口避免全局FFT的边界效应
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x
            
        _, _, H_pad, W_pad = x_pad.shape
        
        # 窗口化FFT
        high_freq_sum = torch.zeros_like(x_pad)
        
        for i in range(0, H_pad, ws):
            for j in range(0, W_pad, ws):
                window = x_pad[:, :, i:i+ws, j:j+ws]
                
                # FFT
                freq = torch.fft.rfft2(window, norm='ortho')
                
                # 高频掩码（抑制低频）
                h_center = ws // 4
                w_center = (ws // 2 + 1) // 4
                
                mask = torch.ones(ws, ws // 2 + 1, device=x.device)
                mask[:h_center, :w_center] = 0.0
                mask[-h_center:, :w_center] = 0.0
                
                high_freq = freq * mask.unsqueeze(0).unsqueeze(0)
                
                # IFFT
                high_spatial = torch.fft.irfft2(high_freq, s=(ws, ws), norm='ortho')
                high_freq_sum[:, :, i:i+ws, j:j+ws] = high_spatial
        
        # 裁剪回原始尺寸
        high_freq_map = high_freq_sum[:, :, :H, :W]
        
        # 边缘增强
        high_feat = self.high_freq_extract(high_freq_map)
        edge_enhanced = self.edge_enhance(high_feat)
        
        # 残差融合
        refined = x + self.high_freq_gain * edge_enhanced
        
        return refined

