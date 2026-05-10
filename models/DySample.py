'''
GitHub: https://github.com/tiny-smart/dysample
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(module, mean=0, std=1, bias=0):
    """
    使用正态分布初始化模块的权重和偏置。
    
    参数:
        module: 要初始化的模块
        mean: 正态分布的均值
        std: 正态分布的标准差
        bias: 偏置的初始值
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    """
    使用常数初始化模块的权重和偏置。
    
    参数:
        module: 要初始化的模块
        val: 权重的初始值
        bias: 偏置的初始值
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    """
    动态采样模块(Dynamic Sampling)
    
    该模块实现了基于内容自适应的特征上采样，比传统的双线性插值或反卷积更加灵活。
    它通过学习偏移量来动态调整采样位置，使上采样更加精确，特别适合分割任务中的细节恢复。
    
    网络支持两种操作模式:
    - 'lp': 先学习偏移量，再进行像素重排(Learn then Pixel shuffle)
    - 'pl': 先像素重排，再学习偏移量(Pixel shuffle then Learn)
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        """
        初始化动态采样模块
        
        参数:
            in_channels: 输入特征的通道数
            scale: 上采样的倍数
            style: 操作模式，'lp'或'pl'
            groups: 分组数，用于分组处理特征
            dyscope: 是否使用动态范围控制
        """
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        # 确保style参数有效
        assert style in ['lp', 'pl']
        # 对'pl'模式的通道数要求
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        # 确保通道数能被分组数整除
        assert in_channels >= groups and in_channels % groups == 0

        # 根据不同模式计算输出通道数
        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups  # 2表示x和y方向的偏移
        else:
            out_channels = 2 * groups * scale ** 2

        # 偏移量预测卷积层
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)  # 使用较小的标准差初始化
        
        # 可选的动态范围控制
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)  # 初始化为0

        # 注册初始位置缓冲区
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        """
        初始化采样位置网格
        
        返回:
            初始化的位置偏移网格
        """
        # 生成[-0.5, 0.5]范围内的均匀分布位置
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        # 构建网格并重塑为所需形状
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        """
        使用计算的偏移量对输入特征进行采样
        
        参数:
            x: 输入特征
            offset: 计算的偏移量
            
        返回:
            采样后的特征图
        """
        B, _, H, W = offset.shape
        # 重塑偏移量为适当的形状
        offset = offset.reshape(B, 2, -1, H, W)
        
        # 生成基础坐标网格
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        
        # 归一化坐标到[-1, 1]范围
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        
        # 使用像素重排和重塑操作调整坐标
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        
        # 使用grid_sample进行双线性采样
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        """
        'lp'模式的前向传播: 先学习偏移量，再进行像素重排
        """
        # 计算偏移量，可选使用动态范围控制
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        # 使用计算的偏移量进行采样
        return self.sample(x, offset)

    def forward_pl(self, x):
        """
        'pl'模式的前向传播: 先像素重排，再学习偏移量
        """
        # 先进行像素重排
        x_ = F.pixel_shuffle(x, self.scale)
        # 计算偏移量，可选使用动态范围控制
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        # 使用计算的偏移量进行采样
        return self.sample(x, offset)

    def forward(self, x):
        """
        根据设定的模式选择前向传播方法
        """
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
