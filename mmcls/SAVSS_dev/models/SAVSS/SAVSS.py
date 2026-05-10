'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

# 导入必要的库和模块
from typing import Sequence  # 用于类型提示
import copy  # 用于深拷贝对象
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_  # 从timm库导入DropPath和权重初始化函数
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"  # 自定义DropPath的字符串表示
from mmcv.cnn import build_norm_layer  # 用于构建标准化层
from mmcv.cnn.utils.weight_init import trunc_normal_  # 截断正态分布初始化
from mmcv.runner.base_module import ModuleList  # 模块列表，类似于nn.ModuleList
from mmcls.models.builder import BACKBONES  # mmcls的骨干网络注册器
from mmcls.models.utils import resize_pos_embed, to_2tuple  # 位置编码调整和尺寸转换工具
from mmcls.models.backbones.base_backbone import BaseBackbone  # 基础骨干网络类
from mmcls.SAVSS_dev.models.modules.patch_embed import ConvPatchEmbed  # 卷积patch嵌入模块
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer  # SAVSS层实现
from models.layers import BottleneckDepthwiseConv

@BACKBONES.register_module()  # 注册为mmcls的骨干网络
class SAVSS(BaseBackbone):
    """
    Direction-aware visual state-space backbone for DFECrack.
    
    It combines Mamba-style state-space modeling with lightweight convolutional
    refinements for efficient crack segmentation.
    """
    # 预定义的架构配置字典，包含不同任务的默认参数
    arch_zoo = {
        'Crack': {  # 裂缝检测架构配置
            'patch_size': 8,  # patch大小
            'embed_dims': 256,  # 嵌入维度
            'num_layers': 4,  # SAVSS层数量
            'num_convs_patch_embed': 2,  # patch嵌入中的卷积层数
            'layers_with_dwconv': [],  # 使用深度可分离卷积的层索引
            'layers_with_freq': [0, 1, 2, 3],  # 全部4层使用频率模块
            # 默认配置：Identity（验证有效，86.48% mIoU）
            'freq_module_types': ['identity', 'identity', 'identity', 'identity'],
            'layer_cfgs': {  # 层配置
                'use_rms_norm': False,  # 是否使用RMS标准化
                'mamba_cfg': {  # Mamba模型配置
                    'd_state': 16,  # 状态维度
                    'expand': 2,  # 扩展比例
                    'conv_size': 7,  # 卷积核大小
                    'dt_init': "random",  # 时间步长初始化方式
                    'conv_bias': True,  # 卷积是否使用偏置
                    'bias': True,  # 是否使用偏置
                    'default_hw_shape': (512 // 8, 512 // 8)  # 默认特征图尺寸
                }
            }
        }
    }

    def __init__(self,
                 img_size=224,         # 输入图像大小
                 in_channels=3,        # 输入通道数(RGB=3)
                 arch=None,            # 架构名称，如'Crack'
                 patch_size=16,        # patch大小
                 embed_dims=192,       # 嵌入维度
                 num_layers=20,        # 层数
                 num_convs_patch_embed=1,  # patch嵌入中的卷积层数
                 with_pos_embed=True,  # 是否使用位置编码
                 out_indices=-1,       # 输出的层索引
                 drop_rate=0.,         # dropout率
                 drop_path_rate=0.,    # drop path率
                 norm_cfg=dict(type='LN', eps=1e-6),  # 标准化层配置
                 final_norm=True,      # 是否在最后添加标准化层
                 interpolate_mode='bicubic',  # 插值模式
                 layer_cfgs=dict(),    # 层配置
                 layers_with_dwconv=[], # 使用深度可分离卷积的层
                 init_cfg=None,        # 初始化配置
                 test_cfg=dict(),      # 测试配置
                 convert_syncbn=False, # 是否转换为同步批量标准化
                 freeze_patch_embed=False,  # 是否冻结patch嵌入层
                 freq_module_types=None,
                 fusion_type='acsf',
                 **kwargs):
        """
        初始化SAVSS骨干网络
        
        参数:
            img_size: 输入图像大小
            in_channels: 输入通道数
            arch: 预定义架构名称，如'Crack'
            patch_size: patch大小，用于图像分块
            embed_dims: 嵌入维度
            num_layers: SAVSS层数量
            num_convs_patch_embed: patch嵌入中的卷积层数
            with_pos_embed: 是否使用位置编码
            out_indices: 输出的层索引，用于多尺度特征
            drop_rate: dropout率
            drop_path_rate: drop path率，用于随机深度
            norm_cfg: 标准化层配置
            final_norm: 是否在最后添加标准化层
            interpolate_mode: 插值模式
            layer_cfgs: 层配置
            layers_with_dwconv: 使用深度可分离卷积的层索引
            init_cfg: 初始化配置
            test_cfg: 测试配置
            convert_syncbn: 是否转换为同步批量标准化
            freeze_patch_embed: 是否冻结patch嵌入层
        """
        super(SAVSS, self).__init__(init_cfg)

        self.test_cfg = test_cfg
        self.img_size = to_2tuple(img_size)  # 确保img_size是二元组(H,W)
        self.convert_syncbn = convert_syncbn
        self.arch = arch

        self.fusion_type = fusion_type

        # 如果没有指定架构，使用传入的参数
        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            self.layers_with_freq = []  # 默认不使用频率模块
            self.freq_module_types = ['identity'] * 4  # 默认类型
            _layer_cfgs = layer_cfgs
        else:
            # 如果指定了架构，从预定义配置中加载参数
            assert self.arch in self.arch_zoo.keys()
            self.embed_dims = self.arch_zoo[self.arch]['embed_dims']
            self.num_layers = self.arch_zoo[self.arch]['num_layers']
            self.patch_size = self.arch_zoo[self.arch]['patch_size']
            self.num_convs_patch_embed = self.arch_zoo[self.arch]['num_convs_patch_embed']
            self.layers_with_dwconv = self.arch_zoo[self.arch]['layers_with_dwconv']
            # 🆕 加载频率模块配置
            self.layers_with_freq = self.arch_zoo[self.arch].get('layers_with_freq', [])
            
            # 🆕 频率模块类型：支持列表（每层不同）或单一类型
            if freq_module_types is not None:
                # 命令行指定的优先
                if isinstance(freq_module_types, str):
                    # 如果是单一字符串，扩展为列表
                    self.freq_module_types = [freq_module_types] * self.num_layers
                else:
                    self.freq_module_types = freq_module_types
            else:
                # 从arch_zoo加载
                arch_types = self.arch_zoo[self.arch].get('freq_module_types', None)
                if arch_types is not None:
                    self.freq_module_types = arch_types
                else:
                    # 兼容旧配置（单一freq_module_type）
                    single_type = self.arch_zoo[self.arch].get('freq_module_type', 'gbc_lfe_seq')
                    self.freq_module_types = [single_type] * self.num_layers
            
            _layer_cfgs = self.arch_zoo[self.arch]['layer_cfgs']

        self.with_pos_embed = with_pos_embed  # 是否使用位置编码
        self.interpolate_mode = interpolate_mode  # 插值模式
        self.freeze_patch_embed = freeze_patch_embed  # 是否冻结patch嵌入层
        _drop_path_rate = drop_path_rate  # 随机深度率

        # 创建卷积Patch嵌入层，将图像分割成patches并进行特征提取
        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,        # 输入通道数
            input_size=img_size,            # 输入图像大小
            embed_dims=self.embed_dims,     # 嵌入维度
            num_convs=self.num_convs_patch_embed,  # 卷积层数量
            patch_size=self.patch_size,     # patch大小
            stride=self.patch_size          # 步长
        )
        # 获取patch分辨率
        self.patch_resolution = self.patch_embed.init_out_size
        # 计算patch总数
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        
        # 如果使用位置编码
        if with_pos_embed:
            # 创建可学习的位置编码参数
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            # 使用截断正态分布初始化位置编码
            trunc_normal_(self.pos_embed, std=0.02)
        
        # 在位置编码后添加dropout层
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # 处理输出索引参数
        if isinstance(out_indices, int):
            out_indices = [out_indices]  # 将单个索引转换为列表
        
        # 确保out_indices是序列类型
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        
        # 处理负索引，并验证索引有效性
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index  # 负索引转为正索引
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # 创建随机深度衰减率序列，从0线性增加到drop_path_rate
        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        # 设置层配置
        self.layer_cfgs = _layer_cfgs
        self.layers = ModuleList()
        
        # 如果layer_cfgs是字典，为每一层创建一个独立的配置副本
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]

        # 创建SAVSS层堆栈
        for i in range(self.num_layers):
            _layer_cfg_i = layer_cfgs[i]
            # 更新层配置，添加嵌入维度和随机深度率
            _layer_cfg_i.update({
                "embed_dims": self.embed_dims,
                "drop_path_rate": dpr[i],
                "layer_idx": i  # 🆕 层索引，用于HAB等层级自适应模块
            })
            # 设置是否使用深度可分离卷积
            if i in self.layers_with_dwconv:
                _layer_cfg_i.update({"with_dwconv": True})
            else:
                _layer_cfg_i.update({"with_dwconv": False})
            # 设置是否使用频率模块（只在指定的层使用）
            current_freq_type = self.freq_module_types[i] if i < len(self.freq_module_types) else 'identity'
            if i in self.layers_with_freq:
                _layer_cfg_i.update({
                    "use_freq_module": True,
                    "freq_module_type": current_freq_type,
                })
            else:
                _layer_cfg_i.update({"use_freq_module": False})
            
            _layer_cfg_i.update({
                "fusion_type": self.fusion_type
            })
            
            # 添加SAVSS层
            self.layers.append(
                SAVSS_Layer(**_layer_cfg_i)  # 创建SAVSS层实例
            )

        # 是否在最后一层添加标准化
        self.final_norm = final_norm
        if final_norm:
            # 创建标准化层
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)  # 添加到模型

        # 为每个输出层添加标准化层
        for i in out_indices:
            if i != self.num_layers - 1:  # 除了最后一层
                if norm_cfg is not None:
                    # 创建标准化层
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    # 如果没有标准化配置，使用恒等映射
                    norm_layer = nn.Identity()
                # 添加标准化层到模型
                self.add_module(f'norm_layer{i}', norm_layer)

        # 创建通道转换模块，用于将256维特征转换为不同通道数
        # 这些模块用于生成多尺度特征金字塔
        
        # 256->128通道转换，用于第一个输出层
        self.conv256to128 = BottleneckDepthwiseConv(in_channels=256, out_channels=128, mid_channels=32,
                                                    kernel_size=1, stride=1, padding=0)
        # 256->64通道转换，用于第二个输出层
        self.conv256to64 = BottleneckDepthwiseConv(in_channels=256, out_channels=64, mid_channels=16,
                                                   kernel_size=1, stride=1, padding=0)
        # 256->32通道转换，用于第三个输出层
        self.conv256to32 = BottleneckDepthwiseConv(in_channels=256, out_channels=32, mid_channels=8,
                                                   kernel_size=1, stride=1, padding=0)
        # 256->16通道转换，用于第四个输出层
        self.conv256to16 = BottleneckDepthwiseConv(in_channels=256, out_channels=16, mid_channels=4,
                                                   kernel_size=1, stride=1, padding=0)
        
        # 为不同通道数创建组标准化层
        self.gn128 = nn.GroupNorm(num_channels=128, num_groups=8)  # 128通道，8组
        self.gn64 = nn.GroupNorm(num_channels=64, num_groups=4)    # 64通道，4组
        self.gn32 = nn.GroupNorm(num_channels=32, num_groups=2)    # 32通道，2组
        self.gn16 = nn.GroupNorm(num_channels=16, num_groups=2)    # 16通道，2组

    @property
    def norm1(self):
        """
        获取norm1标准化层的属性访问器
        
        返回:
            模型中的norm1标准化层
        """
        return getattr(self, self.norm1_name)

    def init_weights(self):
        """
        初始化模型权重
        
        如果不是预训练模型，初始化位置编码
        """
        super(SAVSS, self).init_weights()  # 调用父类的初始化方法
        
        # 如果不是预训练模型，初始化位置编码
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.with_pos_embed:
                # 使用截断正态分布初始化位置编码
                trunc_normal_(self.pos_embed, std=0.02)
        
        # 设置是否冻结patch嵌入层
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        """
        设置是否冻结patch嵌入层
        
        如果freeze_patch_embed为True，将patch_embed设为评估模式，
        并冻结其参数，使其在训练过程中不更新。
        """
        if self.freeze_patch_embed:
            self.patch_embed.eval()  # 设置为评估模式
            for param in self.patch_embed.parameters():
                param.requires_grad = False  # 冻结参数

    def forward(self, x):
        """
        模型前向传播
        
        参数:
            x: 输入图像张量，形状为[B, C, H, W]
            
        返回:
            outs: 多尺度特征列表，用于分割任务
        """
        # 通过patch嵌入层处理输入图像
        x, patch_resolution = self.patch_embed(x)
        
        # 添加位置编码（如果启用）
        if self.with_pos_embed:
            # 调整位置编码大小以匹配当前patch分辨率
            pos_embed = resize_pos_embed(
                self.pos_embed,          # 原始位置编码
                self.patch_resolution,   # 原始patch分辨率
                patch_resolution,        # 当前patch分辨率
                mode=self.interpolate_mode,  # 插值模式
                num_extra_tokens=0       # 额外token数量
            )
            x = x + pos_embed  # 将位置编码添加到特征中
            
        # 应用dropout
        x = self.drop_after_pos(x)

        # 初始化输出列表
        outs_before = []  # 原始输出
        outs = []         # 处理后的输出
        
        # 逐层处理特征
        # ⚠️ 重要：所有4层SAVSS_Layer处理的都是相同分辨率的特征图！
        # 输入512×512，patch_size=8，所以patch_resolution = 64×64
        # 所有layer都处理64×64的特征，只是在输出时才做通道转换和上采样
        for i, layer in enumerate(self.layers):
            # 通过SAVSS层处理特征
            # ⚠️ 注意：hw_shape = patch_resolution = (64, 64)，对所有层都相同！
            x = layer(x, hw_shape=patch_resolution)
            
            # 如果是最后一层且启用了最终标准化
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)  # 应用标准化

            # 如果当前层是指定的输出层
            if i in self.out_indices:
                B, _, C = x.shape
                # 将特征重塑为空间形式
                patch_token = x.reshape(B, *patch_resolution, C)
                
                # 如果不是最后一层，应用对应的标准化
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = norm_layer(patch_token)
                    
                # 调整通道顺序为[B, C, H, W]，适合卷积操作
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs_before.append(patch_token)

                # 根据不同输出层索引，应用不同的通道转换和上采样
                # ⚠️ 重要：虽然所有SAVSS层内部处理的都是64×64，
                # 但输出时通过通道转换+上采样生成多尺度特征金字塔！
                # 这就是图中"DownSample"的实际实现方式（实际是上采样到不同尺寸）
                if i == self.out_indices[0]:
                    # Layer0输出: 64×64 → 256ch转128ch → 上采样到64×64 (对应图中F4: 128×64×64)
                    patch_token_mid = self.gn128(self.conv256to128(patch_token))
                    patch_token_mid = nn.Upsample(size=(64, 64), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[1]:
                    # Layer1输出: 64×64 → 256ch转64ch → 上采样到128×128 (对应图中F3: 64×128×128)
                    patch_token_mid = self.gn64(self.conv256to64(patch_token))
                    patch_token_mid = nn.Upsample(size=(128, 128), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[2]:
                    # Layer2输出: 64×64 → 256ch转32ch → 上采样到256×256 (对应图中F2: 32×256×256)
                    patch_token_mid = self.gn32(self.conv256to32(patch_token))
                    patch_token_mid = nn.Upsample(size=(256, 256), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[3]:
                    # Layer3输出: 64×64 → 256ch转16ch → 上采样到512×512 (对应图中F1: 16×512×512)
                    patch_token_mid = self.gn16(self.conv256to16(patch_token))
                    patch_token_mid = nn.Upsample(size=(512, 512), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                else:
                    continue

        # 返回多尺度特征列表，用于后续的分割头
        return outs
