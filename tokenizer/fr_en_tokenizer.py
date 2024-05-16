import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from .base_tokenizer import BaseTokenizer

class Tokenizer(BaseTokenizer):
    def __init__(self, max_len=256):
        # multi_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

        # self.src_tokenizer = multi_tokenizer
        self.src_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

        # self.tgt_tokenizer = multi_tokenizer
        self.tgt_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

        # check for bos, eos, pad, unk tokens, and add the corresponding tokens
        if self.src_tokenizer.bos_token is None:
            self.src_tokenizer.add_special_tokens({"bos_token": "<bos>"})
        if self.src_tokenizer.eos_token is None:
            self.src_tokenizer.add_special_tokens({"eos_token": "<eos>"})
        if self.src_tokenizer.pad_token is None:
            self.src_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if self.src_tokenizer.unk_token is None:
            self.src_tokenizer.add_special_tokens({"unk_token": "<unk>"})
        
        if self.tgt_tokenizer.bos_token is None:
            self.tgt_tokenizer.add_special_tokens({"bos_token": "<bos>"})
        if self.tgt_tokenizer.eos_token is None:
            self.tgt_tokenizer.add_special_tokens({"eos_token": "<eos>"})
        if self.tgt_tokenizer.pad_token is None:
            self.tgt_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if self.tgt_tokenizer.unk_token is None:
            self.tgt_tokenizer.add_special_tokens({"unk_token": "<unk>"})
        
        self.src_vocab_size = self.src_tokenizer.vocab_size + 4
        self.tgt_vocab_size = self.tgt_tokenizer.vocab_size + 4

        self.max_len = max_len
    
    def tokenize_src(self, src_text):
        # src_text = self.src_tokenizer.bos_token + src_text + self.src_tokenizer.eos_token
        src = self.src_tokenizer.encode(src_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)
        # src = self.pad_or_trunc_tokens(src, self.src_tokenizer.pad_token_id)
        return src
    
    def tokenize_tgt(self, tgt_text):
        # input_tgt_text = self.tgt_tokenizer.bos_token + tgt_text
        # output_tgt_text = tgt_text + self.tgt_tokenizer.eos_token
        tgt_text = self.tgt_tokenizer.bos_token + tgt_text + self.tgt_tokenizer.eos_token
        # input_tgt = self.tgt_tokenizer.encode(input_tgt_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)
        # output_tgt = self.tgt_tokenizer.encode(output_tgt_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)
        tgt = self.tgt_tokenizer.encode(tgt_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)
        # input_tgt = self.pad_or_trunc_tokens(input_tgt, self.tgt_tokenizer.pad_token_id)
        # output_tgt = self.pad_or_trunc_tokens(output_tgt, self.tgt_tokenizer.pad_token_id)
        # return input_tgt, output_tgt
        return tgt

    def decode_output(self, output):
        return self.tgt_tokenizer.decode(output, skip_special_tokens=True)
    
    # def pad_or_trunc_tokens(self, tokens, pad_id): # tokens: torch.Tensor
    #     if tokens.shape[1] < self.max_len:
    #         pad = torch.ones((tokens.shape[0], self.max_len - tokens.shape[1]), dtype=torch.long)
    #         pad = pad * pad_id
    #         tokens = torch.cat((tokens, pad), dim=1)
    #     elif tokens.shape[1] > self.max_len:
    #         tokens = tokens[:, :self.max_len]
    #     return tokens
    