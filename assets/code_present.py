class TranslationTransformer(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: dict, 模型的配置参数，包括：
                src_vocab_size: int, 源语言词表大小
                tgt_vocab_size: int, 目标语言词表大小
                d_model: int, 模型的维度大小
                nhead: int, 多头注意力的头数
                num_encoder_layers: int, 编码器的层数
                num_decoder_layers: int, 解码器的层数
                dim_feedforward: int, 前馈神经网络的隐层维度
                dropout: float, dropout概率
                abs_pos: str, 位置编码的类型，'learnable', 'sinusoidal', 'none'
                qkv_pos: str, QKV的位置编码类型，'rope', 'none'
                seq_len: int, 序列的最长长度
        """

        ...
        if config['abs_pos'] == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(config['d_model'], config['seq_len'])
        elif config['abs_pos'] == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(config['d_model'], config['seq_len'])
        elif config['abs_pos'] == 'none':
            self.positional_encoding = nn.Identity()
        ...

    def forward(self, src, tgt, masks):
        """
        Args:
            src: Tensor, 源语言序列，[batch, seq_len]
            tgt: Tensor, 目标语言序列，[batch, seq_len]
            masks: dict, 包括：
                src_mask: Tensor, Encoder Self-Attention 的 mask，[batch, seq_len, seq_len]
                tgt_mask: Tensor, Decoder Self-Attention 的 mask，[batch, seq_len, seq_len]
                memory_mask: Tensor, Decoder-Encoder Cross-Attention 的 mask，[batch, seq_len, seq_len]
                src_key_padding_mask: Tensor, Encoder Self-Attention 的 key padding mask，[batch, seq_len]
                tgt_key_padding_mask: Tensor, Decoder Self-Attention 的 key padding mask，[batch, seq_len]
        Returns:
            output: Tensor, 模型的输出，[batch, seq_len, tgt_vocab_size]
        """

        src = self.src_word_embedding(src)  # [batch, seq_len] -> [batch, seq_len, d_model]
        src = self.positional_encoding(src)  # [batch, seq_len, d_model]
        src = self.dropout(src)  # [batch, seq_len, d_model]

        tgt = self.tgt_word_embedding(tgt)  # [batch, seq_len] -> [batch, seq_len, d_model]
        tgt = self.positional_encoding(tgt)  # [batch, seq_len, d_model]
        tgt = self.dropout(tgt)  # [batch, seq_len, d_model]

        # output = self.model(src, tgt, **masks)  # -> [batch, seq_len, d_model]
        output = self.model(src, tgt, masks)

        output = self.fc(output)  # [batch, seq_len, d_model] -> [batch, seq_len, tgt_vocab_size]

        return output  # [batch, seq_len, tgt_vocab_size]

    def encoder(self, src):
        """
        Args:
            src: Tensor, 源语言序列，[batch, seq_len]
        Returns:
            memory: Tensor, 编码器的输出，[batch, seq_len, d_model]

        TODO: add mask (由于本任务的翻译过程不需要mask，所以这里暂时没有实现）
        """

        ...
    
    def decoder(self, tgt, memory):
        """
        Args:
            tgt: Tensor, 目标语言序列，[batch, seq_len]
            memory: Tensor, 编码器的输出，[batch, seq_len, d_model]
        Returns:
            output: Tensor, 解码器的输出，[batch, seq_len, tgt_vocab_size]

        TODO: add mask（同上）
        """

        ...


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, qkv_pos='none', seq_len=256
                 ):
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

        ...

        self.qkv_pos = qkv_pos
        if qkv_pos == 'rope':
            freqs_cis = precompute_freqs_cis(d_model, seq_len*2)
            self.register_buffer('freqs_cis', freqs_cis)
        else:
            self.freqs_cis = None

        ...
    
    """
    some other methods
    """

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
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask, freqs_cis=self.freqs_cis)
        return output
    
    """
    some other methods
    """

"""
Encoder, Decoder, EncoderLayer, DecoderLayer, MultiheadAttention Modules

pass freqs_cis all the way down to the multi-head attention module
"""

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
    k = k_proj(key)
    v = v_proj(value)

    if qkv_pos == 'rope':
        assert freqs_cis is not None
        q,k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
    
    ...
    
    # (Masked) Multi-head Attention
