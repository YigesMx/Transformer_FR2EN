import torch
import torch.nn as nn
# import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # buffer是模型的一部分，但不是模型的参数，不会被优化器更新

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:, :x.size(1)]
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_positional_embedding=512):
        super(LearnablePositionalEncoding, self).__init__()

        self.learnable_pos = nn.Embedding(max_positional_embedding, d_model)
        pos_ids = torch.arange(max_positional_embedding).unsqueeze(0)
        self.register_buffer('pos_ids', pos_ids)

    def forward(self, x):
        pos = self.learnable_pos(self.pos_ids[:, :x.size(1)])
        x = x + pos
        return x

# 生成旋转角度
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis[: xq.shape[1]]).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis[: xk.shape[1]]).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def sin_position_encoding(batch_size, max_len, output_dim):
    '''
    :return: [batch_size, max_len, d_model]
    '''
    pe = torch.zeros(max_len, output_dim)  # [max_len, d_model]
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
    div_term = torch.exp(torch.arange(0, output_dim, 2).float() * (-math.log(10000.0) / output_dim))  # [d_model/2]
    pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(1)
    pe = pe.transpose(0, 1)
    return pe


def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size_q = q.shape[0]
    max_len_q = q.shape[1]
    output_dim_q = q.shape[-1]

    batch_size_k = k.shape[0]
    max_len_k = k.shape[1]
    output_dim_k = k.shape[-1]

    # (bs, head, max_len, output_dim)
    '''
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)
    '''
    # (bs, max_len_q/k, output_dim_q/k)
    pos_emb_q = sin_position_encoding(batch_size=batch_size_q, max_len=max_len_q, output_dim=output_dim_q)
    pos_emb_k = sin_position_encoding(batch_size=batch_size_k, max_len=max_len_k, output_dim=output_dim_k)

    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos_q = pos_emb_q[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos_q = pos_emb_q[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    cos_pos_k = pos_emb_k[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos_k = pos_emb_k[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

    # 更新qw, *对应位置相乘
    q = q * cos_pos_q + q2 * sin_pos_q
    
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos_k + k2 * sin_pos_k

    return q, k