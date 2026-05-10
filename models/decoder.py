'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch
from torch import nn
from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.frequency_decoder import build_frequency_decoder


class Decoder(nn.Module):
    """
    DFECrack segmentation model.
    
    网络结构:
    1. 使用方向感知状态空间骨干网络提取多尺度特征
    2. 通过频率增强解码头生成最终分割结果
    """
    def __init__(self, backbone, args=None, decoder_type='fasf_simple'):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.decoder_type = decoder_type
        
        embedding_dim = 8
        self.segmentation_head = build_frequency_decoder(embedding_dim, decoder_type)

    def forward(self, samples):
        features = self.backbone(samples)
        out = self.segmentation_head(features)
        return out


class DiceLoss(nn.Module):
    """Dice损失函数"""
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


class bce_dice(nn.Module):
    """组合BCE和Dice损失的混合损失函数"""
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice


class bce_dice_boundary(nn.Module):
    """
    BCE + Dice + Boundary 边界感知损失
    
    在原始损失基础上，对边界区域增加权重
    """
    def __init__(self, args):
        super(bce_dice_boundary, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_fn = DiceLoss()
        self.args = args
        self.boundary_boost = getattr(args, 'boundary_boost', 2.0)
        
        # 边界提取核
        self.kernel_size = 3
        
    def _extract_boundary(self, mask):
        """从mask中提取边界"""
        # 形态学梯度：膨胀 - 腐蚀
        import torch.nn.functional as F
        dilated = F.max_pool2d(mask, self.kernel_size, stride=1, padding=self.kernel_size//2)
        eroded = -F.max_pool2d(-mask, self.kernel_size, stride=1, padding=self.kernel_size//2)
        boundary = torch.clamp(dilated - eroded, 0, 1)
        return boundary

    def forward(self, y_pred, y_true):
        # 提取边界
        boundary = self._extract_boundary(y_true)
        
        # BCE（边界加权）
        pixel_bce = self.bce_fn(y_pred, y_true)
        weight = 1.0 + (self.boundary_boost - 1.0) * boundary
        weighted_bce = (pixel_bce * weight).mean()
        
        # Dice
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        
        return self.args.BCELoss_ratio * weighted_bce + self.args.DiceLoss_ratio * dice


def build(args):
    """
    构建完整的分割模型和损失函数。
    
    参数:
        args: 包含模型配置的参数对象
        
    返回:
        model: 构建好的Decoder模型
        criterion: 配置好的损失函数
    """
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    # 解析频率模块类型（支持列表，每层不同）
    # 格式: "identity,identity,identity,identity" 或单一类型 "identity"
    freq_module_types = None
    if hasattr(args, 'freq_module_types') and args.freq_module_types:
        if ',' in args.freq_module_types:
            freq_module_types = [x.strip() for x in args.freq_module_types.split(',')]
        else:
            freq_module_types = args.freq_module_types
    
    fusion_type = getattr(args, 'fusion_type', 'acsf')

    backbone = SAVSS(
        arch='Crack',
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.2,
        final_norm=True,
        convert_syncbn=True,
        freq_module_types=freq_module_types,
        fusion_type=fusion_type,
    )
    
    # 解析解码器类型
    decoder_type = getattr(args, 'decoder_type', 'original')
    
    # 构建解码器模型
    model = Decoder(backbone, args, decoder_type=decoder_type)
    
    use_boundary = getattr(args, 'use_boundary', False)
    if use_boundary:
        criterion = bce_dice_boundary(args)
        print(f"Using Boundary-aware Loss (boundary_boost={getattr(args, 'boundary_boost', 2.0)})")
    else:
        criterion = bce_dice(args)
    criterion.to(device)

    return model, criterion
