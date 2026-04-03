

import torch
from torch import nn

class Attention(nn.Module):
    """
    一个简化版的 DSTFormer 中的注意力模块，
    输入 tensor 的形状为 (B, T, J, C)，其中 T 表示时间帧数，J 表示关节数。
    """
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.1, proj_drop=0.1, mode='spatial',  layer_scale_init_value=1.0):
        """
        参数:
            dim_in: 输入特征维度
            dim_out: 输出特征维度
            num_heads: 注意力头数
            qkv_bias: 是否在 qkv 层使用偏置
            qk_scale: 缩放因子
            attn_drop: 注意力 dropout 概率
            proj_drop: 输出投影 dropout 概率
            mode: 'spatial' 或 'temporal'
            n_frames: 时间帧数，根据外部数据传入（必须提供）
        """
        super().__init__()
            
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # 将 attn_weight 扩展为多头权重，形状为 (num_heads,)
        self.attn_weight = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)

    def forward(self, x):
        """
        输入:
            x: (B, T, J, C)
        输出:
            x: (B, T, J, dim_out)
        """
        B, T, J, C = x.shape

        # 计算 q, k, v 并调整形状为 (3, B, num_heads, T, J, C_head)
        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, num_heads, T, J, C_head)
        
        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v)
        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        """
        spatial 模式下的注意力计算，输入 q 的形状为 (B, num_heads, T, J, C_head)
        输出 x 的形状为 (B, T, J, C)
        """
        B, H, T, J, C = q.shape
        # 计算注意力权重：(B, H, T, J, J)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 应用多头注意力权重
        attn = attn * self.attn_weight.view(1, H, 1, 1, 1)  # 每头独立缩放 消融1

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 得到加权后的输出：(B, H, T, J, C)
        x = attn @ v
        # 调整维度，reshape 到 (B, T, J, C * num_heads)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)

    
        return x  # (B, T, J, C)

    def forward_temporal(self, q, k, v):
        """
        temporal 模式下的注意力计算，输入 q 的形状为 (B, num_heads, T, J, C_head)
        输出 x 的形状为 (B, T, J, C)
        """
        B, H, T, J, C = q.shape
        # 转置时间和关节维度，调整为 (B, H, J, T, C)
        qt = q.transpose(2, 3)
        kt = k.transpose(2, 3)
        vt = v.transpose(2, 3)

        # 计算注意力权重：(B, H, J, T, T)
        attn = (qt @ kt.transpose(-2, -1)) * self.scale

        # 应用多头注意力权重
        attn = attn * self.attn_weight.view(1, H, 1, 1, 1)  # 每头独立缩放  消融2

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 得到加权输出：(B, H, J, T, C)
        x = attn @ vt
        # 调整维度，reshape 到 (B, T, J, C * num_heads)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)

       
        return x  # (B, T, J, C)

    



