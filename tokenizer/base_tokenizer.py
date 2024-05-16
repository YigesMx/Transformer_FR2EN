from abc import ABC, abstractmethod

import torch
from torch.nn.utils.rnn import pad_sequence

class BaseTokenizer(ABC):
    @abstractmethod
    def __init__(self, max_len=256):
        raise NotImplementedError
    
    @abstractmethod
    def tokenize_src(self, src_text):
        raise NotImplementedError
    
    @abstractmethod
    def tokenize_tgt(self, tgt_text):
        raise NotImplementedError

    @abstractmethod
    def decode_output(self, output):
        raise NotImplementedError

    def pad_batch(self, batch_tokens) -> torch.Tensor:
        return pad_sequence(batch_tokens, batch_first=True, padding_value=self.src_tokenizer.pad_token_id)