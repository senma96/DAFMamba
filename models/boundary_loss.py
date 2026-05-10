"""
Boundary Supervision Module for Crack Segmentation

边界监督模块

核心思想：
1. 裂缝是细长结构，边界 ≈ 前景本身
2. 边界监督强化模型对裂缝边缘的学习
3. 和 FASF 的高频特征形成呼应

Author: Hui Liu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryExtractor(nn.Module):
    """
    从 Ground Truth 中提取边界
    
    方法：使用 Laplacian 或形态学梯度提取边界
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Laplacian 卷积核（边缘检测）
        laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)
        
    def forward(self, mask):
        """
        Args:
            mask: [B, 1, H, W] 二值掩码
        Returns:
            boundary: [B, 1, H, W] 边界图
        """
        # 方法1: 形态学梯度（膨胀 - 腐蚀）
        dilated = F.max_pool2d(mask, self.kernel_size, stride=1, padding=self.kernel_size//2)
        eroded = -F.max_pool2d(-mask, self.kernel_size, stride=1, padding=self.kernel_size//2)
        boundary = dilated - eroded
        
        # 归一化到 [0, 1]
        boundary = torch.clamp(boundary, 0, 1)
        
        return boundary


class BoundaryHead(nn.Module):
    """
    边界预测头
    
    从特征图预测边界
    """
    def __init__(self, in_channels, mid_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1)  # 输出边界预测
        )
        
    def forward(self, x):
        return self.conv(x)


class BoundaryLoss(nn.Module):
    """
    边界监督损失
    
    计算预测边界和真实边界之间的损失
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred_boundary, gt_boundary):
        """
        Args:
            pred_boundary: [B, 1, H, W] 预测边界 (logits)
            gt_boundary: [B, 1, H, W] 真实边界
        Returns:
            loss: 边界损失
        """
        return self.weight * self.bce(pred_boundary, gt_boundary)


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失（不需要额外预测头）
    
    在原始分割损失上增加边界区域的权重
    
    设计原理：
    1. 提取GT的边界区域
    2. 边界区域的损失权重更高
    3. 让模型更关注边界学习
    """
    def __init__(self, boundary_weight=2.0, kernel_size=3):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_extractor = BoundaryExtractor(kernel_size)
        self.bce_fn = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] 预测 (logits)
            target: [B, 1, H, W] 真实标签
        Returns:
            loss: 加权后的损失
        """
        # 提取边界
        boundary = self.boundary_extractor(target)  # [B, 1, H, W]
        
        # 计算逐像素 BCE
        bce_loss = self.bce_fn(pred, target)  # [B, 1, H, W]
        
        # 边界区域权重更高
        weight = 1.0 + (self.boundary_weight - 1.0) * boundary
        
        # 加权平均
        weighted_loss = (bce_loss * weight).mean()
        
        return weighted_loss


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


class BCEDiceBoundaryLoss(nn.Module):
    """
    BCE + Dice + Boundary 联合损失
    
    组合三种损失：
    1. BCE: 像素级分类
    2. Dice: 区域重叠
    3. Boundary: 边界精度
    
    公式: L = λ1*BCE + λ2*Dice + λ3*Boundary
    """
    def __init__(self, bce_weight=0.83, dice_weight=0.17, boundary_weight=0.5,
                 boundary_type='aware'):
        """
        Args:
            bce_weight: BCE损失权重
            dice_weight: Dice损失权重
            boundary_weight: 边界损失权重
            boundary_type: 边界损失类型
                - 'aware': 边界感知加权（不需要额外预测头）
                - 'explicit': 显式边界预测（需要额外预测头）
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_type = boundary_type
        
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.boundary_extractor = BoundaryExtractor()
        
        if boundary_type == 'aware':
            self.boundary_bce = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.boundary_bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target, pred_boundary=None):
        """
        Args:
            pred: [B, 1, H, W] 分割预测 (logits)
            target: [B, 1, H, W] 真实标签
            pred_boundary: [B, 1, H, W] 边界预测 (logits)，仅 boundary_type='explicit' 时需要
        Returns:
            loss: 总损失
            loss_dict: 各部分损失（用于日志）
        """
        # 1. BCE 损失
        bce_loss = self.bce_fn(pred, target)
        
        # 2. Dice 损失
        dice_loss = self.dice_fn(pred.sigmoid(), target)
        
        # 3. 边界损失
        gt_boundary = self.boundary_extractor(target)
        
        if self.boundary_type == 'aware':
            # 边界感知加权：在边界区域增加损失权重
            pixel_bce = self.boundary_bce(pred, target)
            weight = 1.0 + self.boundary_weight * gt_boundary
            boundary_loss = (pixel_bce * weight).mean() - bce_loss  # 额外的边界损失
        else:
            # 显式边界预测
            if pred_boundary is None:
                raise ValueError("pred_boundary required for explicit boundary type")
            boundary_loss = self.boundary_bce(pred_boundary, gt_boundary)
        
        # 总损失
        total_loss = (self.bce_weight * bce_loss + 
                      self.dice_weight * dice_loss + 
                      self.boundary_weight * boundary_loss)
        
        loss_dict = {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'boundary': boundary_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class SimpleBoundaryLoss(nn.Module):
    """
    简化版边界感知损失（推荐）
    
    在原始 BCE+Dice 基础上，对边界区域增加权重
    不需要修改模型结构，只需要替换损失函数
    
    优点：
    1. 实现简单
    2. 不增加模型参数
    3. 训练稳定
    """
    def __init__(self, bce_weight=0.83, dice_weight=0.17, 
                 boundary_boost=2.0, kernel_size=3):
        """
        Args:
            bce_weight: BCE损失权重
            dice_weight: Dice损失权重
            boundary_boost: 边界区域的损失放大倍数
            kernel_size: 边界提取的核大小
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_boost = boundary_boost
        
        self.bce_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_fn = DiceLoss()
        self.boundary_extractor = BoundaryExtractor(kernel_size)
        
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] 预测 (logits)
            target: [B, 1, H, W] 真实标签
        Returns:
            loss: 总损失
        """
        # 提取边界
        boundary = self.boundary_extractor(target)
        
        # BCE（边界加权）
        pixel_bce = self.bce_fn(pred, target)
        weight = 1.0 + (self.boundary_boost - 1.0) * boundary
        weighted_bce = (pixel_bce * weight).mean()
        
        # Dice
        dice_loss = self.dice_fn(pred.sigmoid(), target)
        
        # 总损失
        total_loss = self.bce_weight * weighted_bce + self.dice_weight * dice_loss
        
        return total_loss


def build_criterion(args):
    """
    构建损失函数
    
    Args:
        args: 参数，包含：
            - BCELoss_ratio: BCE权重
            - DiceLoss_ratio: Dice权重
            - use_boundary: 是否使用边界监督
            - boundary_boost: 边界权重放大倍数
    """
    use_boundary = getattr(args, 'use_boundary', False)
    
    if use_boundary:
        boundary_boost = getattr(args, 'boundary_boost', 2.0)
        return SimpleBoundaryLoss(
            bce_weight=args.BCELoss_ratio,
            dice_weight=args.DiceLoss_ratio,
            boundary_boost=boundary_boost
        )
    else:
        # 原始 BCE+Dice
        from models.decoder import bce_dice
        return bce_dice(args)


