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
        
