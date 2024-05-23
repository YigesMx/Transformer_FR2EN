import torch
import torch.nn as nn
# from torch.nn import Transformer
from .transformer_backend import Transformer

from .positional_encodings import LearnablePositionalEncoding, SinusoidalPositionalEncoding


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
        super(TranslationTransformer, self).__init__()

        assert config['src_vocab_size'] is not None
        assert config['tgt_vocab_size'] is not None
        assert config['d_model'] % config['nhead'] == 0  # d_model必须是nhead的整数倍

        self.src_word_embedding = nn.Embedding(
            num_embeddings=config['src_vocab_size'],
            embedding_dim=config['d_model']
        )

        self.tgt_word_embedding = nn.Embedding(
            num_embeddings=config['tgt_vocab_size'],
            embedding_dim=config['d_model']
        )

        if config['abs_pos'] == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(config['d_model'], config['seq_len'])
        elif config['abs_pos'] == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(config['d_model'], config['seq_len'])
        elif config['abs_pos'] == 'none':
            self.positional_encoding = nn.Identity()

        self.dropout = nn.Dropout(config['dropout'])

        # # torch.nn.Transformer
        # self.model = Transformer(
        #     d_model=config['d_model'],
        #     nhead=config['nhead'],
        #     num_encoder_layers=config['num_encoder_layers'],
        #     num_decoder_layers=config['num_decoder_layers'],
        #     dim_feedforward=config['dim_feedforward'],
        #     dropout=config['dropout'],
        #     batch_first=True
        # )

        # 自定义Transformer
        self.model = Transformer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            qkv_pos=config['qkv_pos'],
            seq_len=config['seq_len']
        )

        self.fc = nn.Linear(config['d_model'], config['tgt_vocab_size'])
    
    def custom_model(self, src, tgt, masks): # 自定义 Transformer 需要交换 seq_len 和 batch_size 的维度 （custom_* 均为用于交换维度并运行模型）
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        output = self.model(src, tgt, **masks)
        output = output.permute(1, 0, 2)
        return output
    
    def cuntom_encoder(self, src):
        src = src.permute(1, 0, 2)
        memory = self.model.encoder(src, freqs_cis=self.model.freqs_cis)
        return memory
    
    def custom_decoder(self, tgt, memory):
        tgt = tgt.permute(1, 0, 2)
        output = self.model.decoder(tgt, memory, freqs_cis=self.model.freqs_cis)
        output = output.permute(1, 0, 2)
        return output

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
        output = self.custom_model(src, tgt, masks)

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
        src = self.src_word_embedding(src)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        # memory = self.model.encoder(src)
        memory = self.cuntom_encoder(src)
        return memory
    
    def decoder(self, tgt, memory):
        """
        Args:
            tgt: Tensor, 目标语言序列，[batch, seq_len]
            memory: Tensor, 编码器的输出，[batch, seq_len, d_model]
        Returns:
            output: Tensor, 解码器的输出，[batch, seq_len, tgt_vocab_size]

        TODO: add mask（同上）
        """
        tgt = self.tgt_word_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        # output = self.model.decoder(tgt, memory)
        output = self.custom_decoder(tgt, memory)
        return output