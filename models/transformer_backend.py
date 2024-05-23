import torch
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
import copy

from .positional_encodings import precompute_freqs_cis, apply_rotary_emb

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, qkv_pos='none', seq_len=256
                 ):
        super(Transformer, self).__init__()

        """
        Args:
            d_model: int, 模型的维度大小
            nhead: int, 多头注意力的头数
            num_encoder_layers: int, 编码器的层数
            num_decoder_layers: int, 解码器的层数
            dim_feedforward: int, 前馈神经网络的隐层维度
            dropout: float, dropout概率
            qkv_pos: str, QKV的位置编码类型，'rope', 'none'
            seq_len: int, 序列的最长长度
        """

        #  ================ 编码部分 =====================
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, qkv_pos=qkv_pos)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # ================ 解码部分 =====================
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, qkv_pos=qkv_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.qkv_pos = qkv_pos
        if qkv_pos == 'rope':
            freqs_cis = precompute_freqs_cis(d_model, seq_len*2)
            self.register_buffer('freqs_cis', freqs_cis)
        else:
            self.freqs_cis = None

        if qkv_pos == 'relative':
            pass

    def _reset_parameters(self):
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src: Tensor, 源语言序列，[src_len, batch_size, embed_dim]
            tgt: Tensor, 目标语言序列，[tgt_len, batch_size, embed_dim]
            src_mask: Tensor, Encoder Self-Attention 的 mask，[src_len, src_len]
            tgt_mask: Tensor, Decoder Self-Attention 的 mask，[tgt_len, tgt_len]
            memory_mask: Tensor, Decoder-Encoder Cross-Attention 的 mask，[tgt_len, src_len]
            src_key_padding_mask: Tensor, Encoder Self-Attention 的 key padding mask，[batch_size, src_len]
            tgt_key_padding_mask: Tensor, Decoder Self-Attention 的 key padding mask，[batch_size, tgt_len]
            memory_key_padding_mask: Tensor, Decoder-Encoder Cross-Attention 的 key padding mask，[batch_size, src_len]
        Returns:
            output: Tensor, 模型的输出，[tgt_len, batch_size, embed_dim]
        """
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, freqs_cis=self.freqs_cis)
        # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask, freqs_cis=self.freqs_cis)
        return output  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]

    # def generate_square_subsequent_mask(self, sz):
    #     r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    #         Unmasked positions are filled with float(0.0).
    #     """
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask  # [sz,sz]

def _get_clones(module, N):
    """
    克隆多个相同的层

    Args:
        module: nn.Module, 一个层
        N: int, 克隆的数量
    Returns:
        nn.ModuleList, 克隆的多个层
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, qkv_pos='none'):
        super(TransformerEncoderLayer, self).__init__()
        """
        Args:
            d_model: int, 模型的维度大小
            nhead: int, 多头注意力的头数
            dim_feedforward: int, 前馈神经网络的隐层维度
            dropout: float, dropout概率
            qkv_pos: str, QKV的位置编码类型，'rope', 'none'
        """
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, qkv_pos=qkv_pos)

        # Implementation of Feedforward model
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, freqs_cis=None):
        """
        Args:
            src: Tensor, 编码部分的输入，[src_len, batch_size, embed_dim]
            src_mask: Tensor, Encoder Self-Attention 的 mask，[src_len, src_len]
            src_key_padding_mask: Tensor, Encoder Self-Attention 的 key padding mask，[batch_size, src_len]
            freqs_cis: Tensor, 旋转位置编码，[seq_len, dim]
        Returns:
            src: Tensor, 编码部分的输出，[src_len, batch_size, embed_dim]
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, freqs_cis=freqs_cis
                              )[0]  # 计算多头注意力
        # src2: [src_len,batch_size,num_heads*kdim] num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # [src_len,batch_size,num_heads*kdim]

        src2 = self.activation(self.linear1(src))  # [src_len,batch_size,dim_feedforward]
        src2 = self.linear2(self.dropout(src2))  # [src_len,batch_size,num_heads*kdim]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        """
        Args:
            encoder_layer: nn.Module, 编码层
            num_layers: int, 编码层的数量
            norm: nn.Module, 归一化层
        """
        self.layers = _get_clones(encoder_layer, num_layers)  # 克隆得到多个encoder layers 论文中默认为6
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, freqs_cis=None):
        """
        Args:
            src: Tensor, 编码部分的输入，[src_len, batch_size, embed_dim]
            mask: Tensor, 编码部分输入的padding情况，[batch_size, src_len]
            src_key_padding_mask: Tensor, 编码部分输入的padding情况，[batch_size, src_len]
            freqs_cis: Tensor, 旋转位置编码，[seq_len, dim]
        Returns:
            output: Tensor, 编码部分的输出，[src_len, batch_size, embed_dim]
        """
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask, freqs_cis=freqs_cis)  # 多个encoder layers层堆叠后的前向传播过程
        if self.norm is not None:
            output = self.norm(output)
        return output  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, qkv_pos='none'):
        super(TransformerDecoderLayer, self).__init__()
        """
        Args:
            d_model: int, 模型的维度大小
            nhead: int, 多头注意力的头数
            dim_feedforward: int, 前馈神经网络的隐层维度
            dropout: float, dropout概率
            qkv_pos: str, QKV的位置编码类型，'rope', 'none'
        """
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, qkv_pos=qkv_pos)
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head attention)
        self.multihead_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, qkv_pos=qkv_pos)
        # 编码部分输出（memory）和解码部分之间的多头注意力机制。
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, freqs_cis=None):
        """
        Args:
            tgt: Tensor, 解码部分的输入，[tgt_len, batch_size, embed_dim]
            memory: Tensor, 编码部分的输出，[src_len, batch_size, embed_dim]
            tgt_mask: Tensor, Decoder Self-Attention 的 mask，[tgt_len, tgt_len]
            memory_mask: Tensor, Decoder-Encoder Cross-Attention 的 mask，[tgt_len, src_len]
            tgt_key_padding_mask: Tensor, Decoder Self-Attention 的 key padding mask，[batch_size, tgt_len]
            memory_key_padding_mask: Tensor, Decoder-Encoder Cross-Attention 的 key padding mask，[batch_size, src_len]
            freqs_cis: Tensor, 旋转位置编码，[seq_len, dim]
        Returns:
            tgt: Tensor, 解码部分的输出，[tgt_len, batch_size, embed_dim]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt,  # [tgt_len,batch_size, embed_dim]
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask, freqs_cis=freqs_cis)[0]
        # 解码部分输入序列之间'的多头注意力（也就是论文结构图中的Masked Multi-head attention)

        tgt = tgt + self.dropout1(tgt2)  # 接着是残差连接
        tgt = self.norm1(tgt)  # [tgt_len,batch_size, embed_dim]

        tgt2 = self.multihead_attn(tgt, memory, memory,  # [tgt_len, batch_size, embed_dim]
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, freqs_cis=freqs_cis)[0]

        # 解码部分的输入经过多头注意力后同编码部分的输出（memory）通过多头注意力机制进行交互
        tgt = tgt + self.dropout2(tgt2)  # 残差连接
        tgt = self.norm2(tgt)  # [tgt_len, batch_size, embed_dim]

        tgt2 = self.activation(self.linear1(tgt))  # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.linear2(self.dropout(tgt2))  # [tgt_len, batch_size, embed_dim]
        # 最后的两层全连接
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, freqs_cis=None):
        """
        Args:
            tgt: Tensor, 解码部分的输入，[tgt_len, batch_size, embed_dim]
            memory: Tensor, 编码部分的输出，[src_len, batch_size, embed_dim]
            tgt_mask: Tensor, Decoder Self-Attention 的 mask，[tgt_len, tgt_len]
            memory_mask: Tensor, Decoder-Encoder Cross-Attention 的 mask，[tgt_len, src_len]
            tgt_key_padding_mask: Tensor, Decoder Self-Attention 的 key padding mask，[batch_size, tgt_len]
            memory_key_padding_mask: Tensor, Decoder-Encoder Cross-Attention 的 key padding mask，[batch_size, src_len]
            freqs_cis: Tensor, 旋转位置编码，[seq_len, dim]
        Returns:
            output: Tensor, 解码部分的输出，[tgt_len, batch_size, embed_dim]
        """
        output = tgt  # [tgt_len,batch_size, embed_dim]

        for mod in self.layers:  # 这里的layers就是N层解码层堆叠起来的
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask, freqs_cis=freqs_cis)
        if self.norm is not None:
            output = self.norm(output)

        return output  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]


class MultiheadAttention(nn.Module):
    r"""
    多头注意力机制，计算公式为：
    ```math
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    ```
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, qkv_pos='none'):
        super(MultiheadAttention, self).__init__()
        """
        Args:
            embed_dim: int, 词嵌入的维度
            num_heads: int, 多头注意力的头数
            dropout: float, dropout概率
            bias: bool, 是否使用偏置
            qkv_pos: str, QKV的位置编码类型，'rope', 'none'
        """
        self.embed_dim = embed_dim  # 前面的d_model参数
        self.head_dim = embed_dim // num_heads  # head_dim 指的就是d_k,d_v
        self.kdim = self.head_dim
        self.vdim = self.head_dim

        self.num_heads = num_heads  # 多头个数
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        # 上面的限制条件就是论文中的  d_k = d_v = d_model/n_head 条件

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # embed_dim = kdim * num_heads
        # 这里第二个维度之所以是embed_dim，实际上这里是同时初始化了num_heads个W_q堆叠起来的, 也就是num_heads个头
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_k,  embed_dim = kdim * num_heads
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_v,  embed_dim = vdim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 最后将所有的Z组合起来的时候，也是一次性完成， embed_dim = vdim * num_heads
        self._reset_parameters()

        self.qkv_pos = qkv_pos

    def _reset_parameters(self):
        """
        初始化参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, freqs_cis=None):
        """
        Args:
            query: Tensor, 查询张量，[tgt_len, batch_size, embed_dim]
            key: Tensor, 键张量，[src_len, batch_size, embed_dim]
            value: Tensor, 值张量，[src_len, batch_size, embed_dim]
            attn_mask: Tensor, 注意力机制的mask，[tgt_len, src_len] or [num_heads*batch_size,tgt_len, src_len]
            key_padding_mask: Tensor, 键的padding mask，[batch_size, src_len]
            freqs_cis: Tensor, 旋转位置编码，[seq_len, dim]
        Returns:
            attn_output: Tensor, 注意力机制的输出，[tgt_len, batch_size, embed_dim]
            attn_output_weights: Tensor, 注意力机制的权重，[batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            out_proj=self.out_proj,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask,
                                            qkv_pos=self.qkv_pos,
                                            freqs_cis=freqs_cis
                                            )
    

def multi_head_attention_forward(query,  # [tgt_len,batch_size, embed_dim]
                                 key,  # [src_len, batch_size, embed_dim]
                                 value,  # [src_len, batch_size, embed_dim]
                                 num_heads,
                                 dropout_p,
                                 out_proj,  # [embed_dim = vdim * num_heads, embed_dim = vdim * num_heads]
                                 training=True,
                                 key_padding_mask=None,  # [batch_size,src_len/tgt_len]
                                 q_proj=None,  # [embed_dim,kdim * num_heads]
                                 k_proj=None,  # [embed_dim, kdim * num_heads]
                                 v_proj=None,  # [embed_dim, vdim * num_heads]
                                 attn_mask=None,  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                                 qkv_pos='none',  # 'rope', 'relative',' none'
                                 freqs_cis=None
                                 ):
    q = q_proj(query)
    #  [tgt_len,batch_size, embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]

    k = k_proj(key)
    # [src_len, batch_size, embed_dim] x [embed_dim, kdim * num_heads] = [src_len, batch_size, kdim * num_heads]

    v = v_proj(value)
    # [src_len, batch_size, embed_dim] x [embed_dim, vdim * num_heads] = [src_len, batch_size, vdim * num_heads]

    if qkv_pos == 'rope':
        q_ = q.transpose(0, 1)
        k_ = k.transpose(0, 1)
        assert freqs_cis is not None
        q,k = apply_rotary_emb(q_, k_, freqs_cis=freqs_cis)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
    elif qkv_pos == 'relative':
        pass
    
    # if is_print_shape:
    #     print("" + "=" * 80)
    #     print("进入多头注意力计算:")
    #     print(
    #         f"\t 多头num_heads = {num_heads}, d_model={query.size(-1)}, d_k = d_v = d_model/num_heads={query.size(-1) // num_heads}")
    #     print(f"\t query的shape([tgt_len, batch_size, embed_dim]):{query.shape}")
    #     print(f"\t  W_q 的shape([embed_dim,kdim * num_heads]):{q_proj.weight.shape}")
    #     print(f"\t   Q  的shape([tgt_len, batch_size,kdim * num_heads]):{q.shape}")
    #     print("\t" + "-" * 70)

    #     print(f"\t  key 的shape([src_len,batch_size, embed_dim]):{key.shape}")
    #     print(f"\t  W_k 的shape([embed_dim,kdim * num_heads]):{k_proj.weight.shape}")
    #     print(f"\t   K  的shape([src_len,batch_size,kdim * num_heads]):{k.shape}")
    #     print("\t" + "-" * 70)

    #     print(f"\t value的shape([src_len,batch_size, embed_dim]):{value.shape}")
    #     print(f"\t  W_v 的shape([embed_dim,vdim * num_heads]):{v_proj.weight.shape}")
    #     print(f"\t   V  的shape([src_len,batch_size,vdim * num_heads]):{v.shape}")
    #     print("\t" + "-" * 70)
    #     print("\t ***** 注意，这里的W_q, W_k, W_v是多个head同时进行计算的. 因此，Q,K,V分别也是包含了多个head的q,k,v堆叠起来的结果 *****")

    tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size, embed_dim]
    src_len = key.size(0)
    head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
    scaling = float(head_dim) ** -0.5
    q = q * scaling  # [query_len,batch_size,kdim * num_heads]

    if attn_mask is not None:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        # 现在 atten_mask 的维度就变成了3D

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads,tgt_len,kdim]
    # 因为前面是num_heads个头一起参与的计算，所以这里要进行一下变形，以便于后面计算。 且同时交换了0，1两个维度
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,kdim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,vdim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]
    # =  [batch_size * num_heads, tgt_len, src_len]  这就num_heads个QK相乘后的注意力矩阵

    if attn_mask is not None:
        attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # 变成 [batch_size, num_heads, tgt_len, src_len]的形状
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'))  #
        # 扩展维度，key_padding_mask从[batch_size,src_len]变成[batch_size,1,1,src_len]
        # 然后再对attn_output_weights进行填充
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,
                                                       src_len)  # [batch_size * num_heads, tgt_len, src_len]

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
    # = # [batch_size * num_heads,tgt_len,vdim]
    # 这就num_heads个Attention(Q,K,V)结果

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # 先transpose成 [tgt_len, batch_size* num_heads ,kdim]
    # 再view成 [tgt_len,batch_size,num_heads*kdim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = out_proj(attn_output)
    # 这里就是多个z  线性组合成Z  [tgt_len,batch_size,embed_dim]
    # if is_print_shape:
    #     print(f"\t 多头注意力中,多头计算结束后的形状（堆叠）为([tgt_len,batch_size,num_heads*kdim]){attn_output.shape}")
    #     print(f"\t 多头计算结束后，再进行线性变换时的权重W_o的形状为([num_heads*vdim, num_heads*vdim  ]){out_proj.weight.shape}")
    #     print(f"\t 多头线性变化后的形状为([tgt_len,batch_size,embed_dim]) {Z.shape}")
    return Z, attn_output_weights.sum(dim=1) / num_heads  # average attention weights over heads
