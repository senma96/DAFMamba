import math
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm
from models.layers import BottleneckDepthwiseConv, LocalContextBlock
from models.FeatureFusion import build_fusion_module


class Identity(nn.Module):
    """Identity mapping used by the released frequency-module configuration."""
    def __init__(self, channels):
        super().__init__()
    def forward(self, x):
        return x


class DirectionAwareScanFusion(nn.Module):
    """
    方向感知扫描融合模块 (Direction-Aware Scan Fusion, DASF)
    
    核心创新点：
    - 不同于ASM直接预测4个权重（黑盒方式）
    - DASF先预测每个位置的局部裂缝方向θ
    - 然后根据方向θ与4个扫描方向的匹配度计算权重（白盒方式）
    
    理论依据：
    - 指纹识别中的方向场理论 (Hong et al., PAMI 1998)
    - RoadTracer的道路方向预测 (CVPR 2018)
    - 裂缝具有局部方向性，与扫描方向匹配时效果更好
    
    工作流程：
    1. 预测每个patch的局部裂缝方向 θ = (cos, sin)
    2. 计算θ与4个扫描方向的内积（匹配度）
    3. 对匹配度做softmax得到权重
    4. 加权融合4个扫描结果
    
    参数说明：
    - d_inner: 内部特征维度 (通常是d_model * expand)
    - n_directions: 扫描方向数量 (默认4)
    - temperature: softmax温度参数，控制权重分布的尖锐程度
    """
    
    def __init__(self, d_inner, n_directions=4, temperature=1.0):
        super().__init__()
        self.n_directions = n_directions
        self.temperature = temperature
        
        # 轻量级方向预测网络
        # 输出2通道：(cos θ, sin θ)
        mid_channels = max(d_inner // 8, 32)
        
        self.direction_predictor = nn.Sequential(
            # 深度可分离卷积：捕获局部空间模式
            nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
            nn.GroupNorm(16, d_inner),
            nn.SiLU(inplace=True),
            
            # 通道压缩
            nn.Conv2d(d_inner, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(inplace=True),
            
            # 预测2通道：(cos θ, sin θ)
            nn.Conv2d(mid_channels, 2, kernel_size=1, bias=True),
        )
        
        # 定义4个SASS扫描方向（单位向量）- 方案B1双向扫描
        # o1: 纯垂直向下 ↓ → 主方向90°（向下）
        # o2: 纯垂直向上 ↑ → 主方向-90°/270°（向上）
        # o3: 纯水平向右 → → 主方向0°（向右）
        # o4: 纯水平向左 ← → 主方向180°（向左）
        # 注意：o1和o2方向相反，o3和o4方向相反，形成双向扫描
        # 对于裂缝方向预测，我们只关心方向，不关心正负（裂缝没有"方向"只有"朝向"）
        # 所以使用绝对值内积，或者这里简化：垂直裂缝匹配o1/o2，水平裂缝匹配o3/o4
        scan_directions = torch.tensor([
            [0.0, 1.0],      # dir1: 90° (cos90, sin90) = (0, 1) 垂直向下
            [0.0, -1.0],     # dir2: -90° (cos-90, sin-90) = (0, -1) 垂直向上
            [1.0, 0.0],      # dir3: 0° (cos0, sin0) = (1, 0) 水平向右
            [-1.0, 0.0],     # dir4: 180° (cos180, sin180) = (-1, 0) 水平向左
        ], dtype=torch.float32)  # [4, 2]
        
        # 注册为buffer，不参与训练但会随模型移动到GPU
        self.register_buffer('scan_directions', scan_directions)
        
        # 初始化：让初始预测接近(1,0)即水平方向，便于稳定训练
        nn.init.zeros_(self.direction_predictor[-1].weight)
        nn.init.zeros_(self.direction_predictor[-1].bias)
        self.direction_predictor[-1].bias.data[0] = 1.0  # cos初始化为1
        
    def forward(self, x_2d, y_scans):
        """
        前向传播
        
        参数:
            x_2d: 2D特征图 [B, C, H, W]，用于预测方向
            y_scans: 4个扫描结果的列表，每个形状为 [B, L, C]
            
        返回:
            y_merged: 加权融合后的特征 [B, L, C]
            direction_field: 预测的方向场 [B, 2, H, W]，用于可视化
            scan_weights: 扫描权重 [B, 4, L]，用于分析
        """
        B, C, H, W = x_2d.shape
        L = H * W
        
        # ========== 步骤1: 预测局部方向 ==========
        # 输出 [B, 2, H, W]，表示每个位置的方向向量 (cos θ, sin θ)
        direction_raw = self.direction_predictor(x_2d)  # [B, 2, H, W]
        
        # L2归一化，确保是单位向量
        direction_field = F.normalize(direction_raw, p=2, dim=1)  # [B, 2, H, W]
        
        # ========== 步骤2: 计算方向匹配度 ==========
        # direction_field: [B, 2, H, W]
        # scan_directions: [4, 2]
        # 匹配度 = 内积 = cos(夹角)
        
        # 重排维度以便计算
        direction_flat = direction_field.view(B, 2, L)  # [B, 2, L]
        direction_flat = direction_flat.permute(0, 2, 1)  # [B, L, 2]
        
        # 计算内积：[B, L, 2] @ [2, 4] = [B, L, 4]
        match_scores = torch.matmul(direction_flat, self.scan_directions.T)  # [B, L, 4]
        match_scores = match_scores.permute(0, 2, 1)  # [B, 4, L]
        
        # ========== 步骤3: 匹配度转换为权重 ==========
        # 使用温度参数控制分布尖锐程度
        # temperature小 → 权重更集中于最匹配的方向
        # temperature大 → 权重更均匀
        scan_weights = F.softmax(match_scores / self.temperature, dim=1)  # [B, 4, L]
        
        # ========== 步骤4: 加权融合 ==========
        y_merged = torch.zeros_like(y_scans[0])  # [B, L, C]
        for i in range(self.n_directions):
            weight_i = scan_weights[:, i, :].unsqueeze(-1)  # [B, L, 1]
            y_merged = y_merged + weight_i * y_scans[i]
        
        return y_merged, direction_field, scan_weights


# 保留旧名称的别名，便于兼容
AdaptiveScanMerge = DirectionAwareScanFusion

class SAVSS_2D(nn.Module):
    """
    SAVSS_2D - 2D结构感知视觉状态空间模型
    
    这是SAVSS的核心计算单元，结合了Mamba状态空间模型和结构感知扫描策略(SASS)。
    该模块通过多方向扫描增强对裂缝拓扑结构的感知能力，并使用瓶颈卷积处理空间信息。
    
    处理流程:
    1. 输入特征通过线性投影和分支
    2. 应用2D卷积捕获局部空间信息
    3. 通过SASS策略生成多个扫描路径
    4. 应用选择性扫描处理特征
    5. 融合多方向扫描结果生成输出特征
    """
    def __init__(
            self,
            d_model,                # 输入特征维度
            d_state=16,             # 状态空间维度
            expand=2,               # 内部特征扩展比例
            dt_rank="auto",         # 时间步长矩阵的秩
            dt_min=0.001,           # 时间步长最小值
            dt_max=0.1,             # 时间步长最大值
            dt_init="random",       # 时间步长初始化方式
            dt_scale=1.0,           # 时间步长缩放因子
            dt_init_floor=1e-4,     # 时间步长初始化下限
            conv_size=7,            # 卷积核大小
            bias=False,             # 是否使用偏置
            conv_bias=False,        # 卷积是否使用偏置
            init_layer_scale=None,  # 层缩放初始值
            default_hw_shape=None,  # 默认高宽形状
            scan_type='dasf',
    ):
        """初始化SAVSS_2D模块"""
        super().__init__()
        # 基本参数设置
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 内部特征维度
        # 自动计算时间步长矩阵的秩
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 扫描相关参数
        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.scan_type = scan_type
        self.n_directions = 4  # 扫描方向数量

        # 层缩放参数
        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        # 输入投影层：将d_model维度投影到d_inner*2维度，用于门控机制
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 确保卷积核大小为奇数
        assert conv_size % 2 == 1
        self.conv2d = BottleneckDepthwiseConv(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            mid_channels=self.d_inner // 16,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        # 激活函数设置
        self.activation = "silu"
        self.act = nn.SiLU()

        # 特征投影层：将d_inner维度投影到dt_rank + d_state*2维度
        # 用于生成时间步长和B、C参数
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        # 时间步长投影层：将dt_rank维度投影回d_inner维度
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # 初始化时间步长参数
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 生成初始时间步长值
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # 初始化状态转移矩阵A
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # 初始化D参数
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # 输出投影层
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
        # 方向相关的B参数
        self.direction_Bs = nn.Parameter(torch.zeros(5, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)  # 使用截断正态分布初始化
        
        # 🆕 自适应扫描权重融合模块 (ASM)
        # 核心创新：根据局部裂缝方向自适应选择扫描权重
        self.adaptive_scan_merge = AdaptiveScanMerge(
            d_inner=self.d_inner,
            n_directions=self.n_directions
        )

    def sass(self, hw_shape):
        """
        结构感知扫描策略(Structure-Aware Scanning Strategy) - 方案B1双向扫描
        
        改进点：使用双向纯方向扫描，让信息双向传递
        - o1: 纯垂直向下 ↓↓↓（逐列从上到下）
        - o2: 纯垂直向上 ↑↑↑（逐列从下到上，o1的反向）
        - o3: 纯水平向右 →→→（逐行从左到右）
        - o4: 纯水平向左 ←←←（逐行从右到左，o3的反向）
        
        理论依据：
        1. 纯方向扫描能让Mamba状态沿裂缝方向完整传递
        2. 双向扫描让信息可以双向传递（参考Vim的双向Mamba）
        3. 垂直裂缝：↓和↑都能完整捕获
        4. 水平裂缝：→和←都能完整捕获
        
        参数:
            hw_shape: 特征图的高宽形状(H, W)
            
        返回:
            三元组: (扫描顺序, 逆扫描顺序, 方向指示)
        """
        H, W = hw_shape  # 特征图高宽
        L = H * W        # 总像素数
        
        # 初始化四种扫描路径的索引列表
        o1, o2, o3, o4 = [], [], [], []
        # 初始化方向列表
        d1, d2, d3, d4 = [], [], [], []
        # 初始化逆索引映射，用于恢复原始顺序
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        # ========== o1: 纯垂直向下 ↓↓↓ ==========
        # 扫描顺序：第0列(0→1→2→3) → 第1列(4→5→6→7) → ...
        # 优势：垂直裂缝从上到下连续扫描
        for j in range(W):  # 逐列
            for i in range(H):  # 从上到下
                idx = i * W + j
                o1_inverse[idx] = len(o1)
                o1.append(idx)
                if i < H - 1:
                    d1.append(4)  # 向下
                else:
                    d1.append(1)  # 换列（向右）
        d1 = [0] + d1[:-1]

        # ========== o2: 纯垂直向上 ↑↑↑ (o1的反向) ==========
        # 扫描顺序：第0列(3→2→1→0) → 第1列(7→6→5→4) → ...
        # 优势：垂直裂缝从下到上连续扫描，与o1互补
        for j in range(W):  # 逐列
            for i in range(H - 1, -1, -1):  # 从下到上（反向）
                idx = i * W + j
                o2_inverse[idx] = len(o2)
                o2.append(idx)
                if i > 0:
                    d2.append(3)  # 向上
                else:
                    d2.append(1)  # 换列（向右）
        d2 = [0] + d2[:-1]

        # ========== o3: 纯水平向右 →→→ ==========
        # 扫描顺序：第0行(0→1→2→3) → 第1行(4→5→6→7) → ...
        # 优势：水平裂缝从左到右连续扫描
        for i in range(H):  # 逐行
            for j in range(W):  # 从左到右
                idx = i * W + j
                o3_inverse[idx] = len(o3)
                o3.append(idx)
                if j < W - 1:
                    d3.append(1)  # 向右
                else:
                    d3.append(4)  # 换行（向下）
        d3 = [0] + d3[:-1]

        # ========== o4: 纯水平向左 ←←← (o3的反向) ==========
        # 扫描顺序：第0行(3→2→1→0) → 第1行(7→6→5→4) → ...
        # 优势：水平裂缝从右到左连续扫描，与o3互补
        for i in range(H):  # 逐行
            for j in range(W - 1, -1, -1):  # 从右到左（反向）
                idx = i * W + j
                o4_inverse[idx] = len(o4)
                o4.append(idx)
                if j > 0:
                    d4.append(2)  # 向左
                else:
                    d4.append(4)  # 换行（向下）
        d4 = [0] + d4[:-1]

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))

    def forward(self, x, hw_shape):
        """
        SAVSS_2D前向传播函数
        
        实现了结构感知视觉状态空间模型的核心计算逻辑，包括:
        1. 特征投影和门控分离
        2. 2D卷积处理
        3. 结构感知扫描
        4. 选择性状态空间计算
        5. 多方向特征融合
        
        参数:
            x: 输入特征，形状为[batch_size, L, d_model]
            hw_shape: 特征图的高宽形状(H, W)
            
        返回:
            处理后的特征
        """
        batch_size, L, _ = x.shape  # 获取输入形状
        H, W = hw_shape  # 特征图高宽
        E = self.d_inner  # 内部特征维度

        # 初始化状态(在此实现中未使用)
        conv_state, ssm_state = None, None
        
        # 见图片：imgs/SAVSS_2D.png
        # 输入投影，生成门控机制的两个分支
        xz = self.in_proj(x)
        
        # 计算状态转移矩阵A
        A = -torch.exp(self.A_log.float())  # 负指数化，确保稳定性

        # 将投影结果分为两部分：x用于主路径，z用于门控
        x, z = xz.chunk(2, dim=-1)
        
        # 将序列特征重塑为2D特征图，并调整通道顺序为[B,C,H,W]
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        # 应用2D瓶颈卷积和激活函数
        x_2d = self.act(self.conv2d(x_2d))
        # 将2D特征图重塑回序列形式
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        # 投影特征以生成SSM参数
        x_dbl = self.x_proj(x_conv)
        # 分离出时间步长dt和SSM参数B、C
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # 投影时间步长
        dt = self.dt_proj(dt)
        # 调整形状以适应selective_scan_fn
        dt = dt.permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        # 确认激活函数类型
        assert self.activation in ["silu", "swish"]

        # 生成结构感知扫描路径和方向，对应图中ss2d
        orders, inverse_orders, directions = self.sass(hw_shape)
        # 根据方向获取对应的B参数
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        # 扩展B参数以匹配批次大小
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in
                        direction_Bs]

        # 对每个扫描路径应用选择性扫描
        y_scan = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),  # 按扫描顺序重排特征
                dt,                                             # 时间步长
                A,                                              # 状态转移矩阵
                (B + dB).contiguous(),                          # 方向感知的B参数
                C,                                              # C参数
                self.D.float(),                                 # D参数
                z=None,                                         # 不使用外部z
                delta_bias=self.dt_proj.bias.float(),           # 时间步长偏置
                delta_softplus=True,                            # 使用softplus激活
                return_last_state=ssm_state is not None,        # 是否返回最终状态
            ).permute(0, 2, 1)[:, inv_order, :]                 # 恢复原始顺序
            for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs)
        ]

        # 🆕 方向感知扫描融合 (DASF)
        # 核心创新：预测局部裂缝方向，根据方向匹配度计算扫描权重
        # 替代原来的等权相加: y = sum(y_scan)
        # 返回值：y_merged(融合结果), direction_field(方向场), scan_weights(权重)
        y_merged, direction_field, scan_weights = self.adaptive_scan_merge(x_2d, y_scan)
        
        # 应用门控机制
        y = y_merged * self.act(z)
        # 输出投影
        out = self.out_proj(y)
        # 应用层缩放(如果启用)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out

class SAVSS_Layer(nn.Module):
    """
    SAVSS_Layer - direction-aware visual state-space layer
    
    The layer combines local crack morphology refinement, direction-aware
    state-space scanning and adaptive channel-spatial fusion.
    """
    def __init__(
            self,
            embed_dims,
            use_rms_norm,
            with_dwconv,
            drop_path_rate,
            mamba_cfg,
            use_freq_module=False,
            freq_module_type='identity',
            layer_idx=0,
            fusion_type='acsf',
            **kwargs,
    ):
        """初始化SAVSS层"""
        super(SAVSS_Layer, self).__init__()
        # 更新Mamba配置，添加模型维度
        mamba_cfg.update({'d_model': embed_dims})
        
        # 根据配置选择标准化层类型
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)  # RMS标准化
        else:
            self.norm = nn.LayerNorm(embed_dims)  # 层标准化
        
        # 是否使用深度可分离卷积
        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            # 创建深度可分离卷积模块
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,      # 输入通道数
                    embed_dims,      # 输出通道数
                    kernel_size=(3, 3),  # 3x3卷积核
                    padding=(1, 1),      # 保持空间维度
                    bias=False,          # 不使用偏置
                    groups=embed_dims    # 分组卷积，每个通道单独卷积
                ),
                nn.BatchNorm2d(embed_dims),  # 批量标准化
                nn.GELU(),                   # GELU激活函数
            )

        # 创建SAVSS_2D模块，核心的状态空间模型
        self.SAVSS_2D = SAVSS_2D(**mamba_cfg)
        
        # 随机深度dropout，用于正则化
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        
        # 线性投影层，用于残差连接
        self.linear_256 = nn.Linear(in_features=256, out_features=256, bias=True)
        
        # 组标准化层
        self.GN_256 = nn.GroupNorm(num_channels=256, num_groups=16)
        
        # Optional frequency module. The released model uses identity here.
        self.use_freq_module = use_freq_module
        if use_freq_module:
            if freq_module_type == 'identity':
                self.freq_module = Identity(embed_dims)
            else:
                raise ValueError(f"Unknown freq_module_type: {freq_module_type}. "
                               f"The released model uses 'identity'.")
        else:
            self.use_freq_module = False
        
        self.local_context = LocalContextBlock(embed_dims)
        
        # ACSF fusion module
        self.fusion_module = build_fusion_module(fusion_type, embed_dims)

    def forward(self, x, hw_shape):
        """
        SAVSS_Layer前向传播函数
        
        实现了SAVSS层的完整处理流程，包括:
        1. 局部上下文处理，增强裂缝形态特征
        2. SAVSS_2D处理，捕获长距离依赖
        3. ACSF模块融合，结合原始特征和处理后特征
        4. 残差连接，保持信息流动性
        
        参数:
            x: 输入特征，形状为[B, L, C]
            hw_shape: 特征图的高宽形状(H, W)
            
        返回:
            处理后的特征
        """
        # 获取输入特征的形状
        B, L, C = x.shape
        # 计算特征图的高宽(假设为正方形)
        H = W = int(math.sqrt(L))
        # 将序列特征重塑为2D特征图[B, C, H, W]
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.use_freq_module:
            # 使用频率增强模块
            x = self.freq_module(x)
        else:
            # Two local-context refinements preserve the released architecture.
            for i in range(2):
                x = self.local_context(x)

        # 将特征重塑回序列形式[B, L, C]
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # 应用标准化、SAVSS_2D和随机深度dropout
        # SAVSS_2D是核心的状态空间模型，用于捕获长距离依赖
        mixed_x = self.drop_path(self.SAVSS_2D(self.norm(x), hw_shape))
        
        # 获取处理后特征的形状
        b, l, c = mixed_x.shape
        h = w = int(math.sqrt(l))
        
        # ACSF: fuse local conv features (base) with global Mamba features (guidance)
        mixed_x = self.fusion_module(
            x.permute(0, 2, 1).reshape(b, c, h, w),
            mixed_x.permute(0, 2, 1).reshape(b, c, h, w)
        )
        
        # 应用组标准化并重塑回序列形式
        mixed_x = self.GN_256(mixed_x).reshape(b, c, h * w).permute(0, 2, 1)

        # 如果启用深度可分离卷积，进一步处理特征
        if self.with_dwconv:
            b, l, c = mixed_x.shape
            h, w = hw_shape
            # 重塑为2D特征图
            mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x = self.local_context(mixed_x)
            # 重塑回序列形式
            mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)

        # 创建残差分支
        # 先应用组标准化和通道变换
        mixed_x_res = self.linear_256(self.GN_256(mixed_x.permute(0, 2, 1)).permute(0, 2, 1))
        
        # 添加残差连接，保持信息流动性
        return mixed_x + mixed_x_res
