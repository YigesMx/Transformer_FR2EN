import torch

class MaskMaker(object):
    def __init__(self, wrapped_tokenizer):
        self.warped_tokenizer = wrapped_tokenizer

    # def generate_square_subsequent_mask(self, sz, device):
    #     mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def __call__(self, *args, **kwds):
        return self.create_masks(*args, **kwds)
    
    def create_tgt_mask(self, tgt):
        # tgt: [batch, seq_len]
        tgt_len = tgt.shape[1]
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        return tgt_mask
    
    def create_src_key_padding_mask(self, src):
        # src: [batch, seq_len]
        src_key_padding_mask = (src == self.warped_tokenizer.src_tokenizer.pad_token_id)
        # src_key_padding_mask: [batch, seq_len]
        return src_key_padding_mask
    
    def create_tgt_key_padding_mask(self, tgt):
        # tgt: [batch, seq_len]
        tgt_key_padding_mask = (tgt == self.warped_tokenizer.tgt_tokenizer.pad_token_id)
        # tgt_key_padding_mask: [batch, seq_len]
        return tgt_key_padding_mask
    
    def create_masks(self, src, tgt):
        src_mask = torch.zeros((src.shape[1], src.shape[1]), device=src.device).type(torch.bool)
        tgt_mask = self.create_tgt_mask(tgt)
        src_key_padding_mask = self.create_src_key_padding_mask(src)
        tgt_key_padding_mask = self.create_tgt_key_padding_mask(tgt)
        return {
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "src_key_padding_mask": src_key_padding_mask,
            "tgt_key_padding_mask": tgt_key_padding_mask
        }
    
def format_shifted_tgt(tgt):
    input_tgt = tgt[:, :-1]
    output_tgt = tgt[:, 1:]
    return input_tgt, output_tgt