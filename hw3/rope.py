import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    RoPE 是旋转位置编码，它通过将输入的稠密向量旋转来稳定训练。
    公式是：
    out = x * cos(theta * position) - x * sin(theta * position)
    Args:
        theta (float): 底数超参数
        d_k (int): 输入的维度，也就是d_model
        max_seq_len (int): 最大序列长度
        device (torch.device): 设备
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        token_positions: (batch_size, seq_len) 每个token的位置信息
    output:
        out: (batch_size, seq_len, d_model) 输出的稠密向量
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.theta: float = theta
        self.d_k : int = d_k
        self.max_seq_len : int = max_seq_len
        self.device = device

        # 计算频率
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2)).float() / self.d_k)
        # 记录每个token的位置信息
        positions = torch.arange(self.max_seq_len)
        # 计算正弦和余弦
        sinusoids = torch.outer(positions, freqs)  # outer是外积，即每个位置都与每个频率相乘 shape: [max_seq_len, d_k//2]
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False)  # 利用register_buffer表示这是固定的，不需要学习
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 确保 token_positions 的数据类型是 long
        token_positions = token_positions.long()

        # 获取 token 位置对应的 sin 和 cos 值
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 进行广播处理：batch_size, seq_len, d_k//2
        cos = cos.unsqueeze(0)  # shape: [1, batch_size, seq_len, d_k//2]
        sin = sin.unsqueeze(0)  # shape: [1, batch_size, seq_len, d_k//2]

        # 分离奇偶维度
        x_1 = x[..., 0::2]  # 偶数维度，shape: [batch_size, seq_len, d_k//2]
        x_2 = x[..., 1::2]  # 奇数维度，shape: [batch_size, seq_len, d_k//2]

        # 旋转操作
        output_1 = x_1 * cos - x_2 * sin  # 对偶数维度进行旋转
        output_2 = x_1 * sin + x_2 * cos  # 对奇数维度进行旋转

        # 将两个旋转后的部分拼接起来
        out = torch.stack([output_1, output_2], dim=-1)  # shape: [batch_size, seq_len, d_k//2, 2]
        
        # 展平最后两个维度：将 d_k//2 和 2 维度展平为 d_k
        out = out.flatten(-2)  # shape: [batch_size, seq_len, d_k]
        
        return out