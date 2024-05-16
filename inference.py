import os
import argparse

import torch
import heapq

from tokenizer.fr_en_tokenizer import Tokenizer

from models.transformer import TranslationTransformer
from utils.helper import MaskMaker, format_shifted_tgt

class PriorityQueue(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []

    def push(self, item):
        if len(self.data) < self.max_size:
            heapq.heappush(self.data, item)
        else:
            heapq.heappushpop(self.data, item)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return str(self.data)

def beam_search(model, src_text, wrapped_tokenizer, max_len=256, beam_size=5, device='cpu'):
    src = wrapped_tokenizer.tokenize_src(src_text).to(device)
    # print(src)
    memory = model.encoder(src).to(device)

    input_tgt = wrapped_tokenizer.tokenize_tgt('')[:,:1].type(torch.long).to(device)
    # print(input_tgt)
    # -> [1, 1] = bos_idx

    beam = PriorityQueue(beam_size)
    beam.push((0, input_tgt))

    done = []

    for i in range(max_len-1):
        new_beam = PriorityQueue(beam_size)
        for score, input_tgt in beam:
            output = model.decoder(input_tgt.type(torch.long), memory)
            # print(i)
            # output: [1, cur_seq_len, dim]
            scores = model.fc(output[:, -1])
            # output[:,-1]: [1, dim]
            # prob: [1, tgt_vocab_size]
            real_probabilities = torch.softmax(scores, dim=-1)
            prob, next_word = torch.topk(real_probabilities, beam_size)
            for j in range(beam_size):
                new_score = score + prob[0][j].item()
                new_input_tgt = torch.cat([input_tgt, torch.tensor([[next_word[0][j].item()]]).type(torch.long).to(device)], dim=-1)
                new_beam.push((new_score, new_input_tgt))
        beam = new_beam
        
        for score, input_tgt in beam:
            if input_tgt[0][-1].item() == wrapped_tokenizer.tgt_tokenizer.eos_token_id:
                done.append((score, input_tgt))
        
        if len(done) >= beam_size:
            break
    
    for score, input_tgt in done:
        print(f'{score:.4f}', wrapped_tokenizer.decode_output(input_tgt[0]))

    return done

def greedy_search(model, src_text, wrapped_tokenizer, max_len=256, device='cpu'):
    src = wrapped_tokenizer.tokenize_src(src_text).to(device)
    # print(src)
    memory = model.encoder(src).to(device)

    input_tgt = wrapped_tokenizer.tokenize_tgt('')[:,:1].type(torch.long).to(device)
    # print(input_tgt)
    # -> [1, 1] = bos_idx

    for i in range(max_len-1):
        output = model.decoder(input_tgt.type(torch.long), memory)
        # print(i)
        # output: [1, cur_seq_len, dim]
        prob = model.fc(output[:, -1])
        # output[:,-1]: [1, dim]
        # prob: [1, tgt_vocab_size]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        input_tgt = torch.cat([input_tgt, torch.tensor([[next_word]]).type(torch.long).to(device)], dim=-1)
        # input_tgt: [1, cur_seq_len+1]
        if next_word == wrapped_tokenizer.tgt_tokenizer.eos_token_id:
            break
    
    print(wrapped_tokenizer.decode_output(input_tgt[0]))

    return input_tgt

def test(model, src_text, tgt_text, wrapped_tokenizer, device='cpu'):
    src = wrapped_tokenizer.tokenize_src(src_text)
    src = src.to(device)
    tgt = wrapped_tokenizer.tokenize_tgt(tgt_text)
    input_tgt, output_tgt = format_shifted_tgt(tgt)
    input_tgt = input_tgt.to(device)
    output_tgt = output_tgt.to(device)
    # -> [1, seq_len]

    mask_maker = MaskMaker(wrapped_tokenizer)

    masks = mask_maker.create_masks(src, input_tgt)

    output = model(src=src, tgt=input_tgt, masks=masks)

    # decode
    output = torch.argmax(output, dim=-1)
    print('output_tokens: ', output)
    print('output_tgt_tokens: ', output_tgt)
    print('output_text:', wrapped_tokenizer.decode_output(output[0]))
    

def arg_parse():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model_path', type=str, default='runs/transformer_512dh8_e6d6_epochbased_rope_20240516_011720/checkpoints/best_checkpoint_checkpoint_19_loss=-1.7915.pt', help='model path')
    parser.add_argument('--config', type=str, default='configs/transformer_512dh8_e6d6_epochbased_rope.py', help='config file')
    parser.add_argument('--src_text', type=str, default="Selon les dispositions de la Charte des Nations Unies, l'admission de nouveaux membres doit être approuvée en même temps par le Conseil de sécurité et l'Assemblée générale.", help='source text')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()

    # config
    from importlib.machinery import SourceFileLoader
    config = SourceFileLoader('config', args.config).load_module()
    model_params = config.model_params
    train_params = config.train_params
    valid_params = config.valid_params

    # device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # tokenizer
    wrapped_tokenizer = Tokenizer(model_params['seq_len'])

    model_params['src_vocab_size'] = wrapped_tokenizer.src_vocab_size
    model_params['tgt_vocab_size'] = wrapped_tokenizer.tgt_vocab_size
    # model
    model = TranslationTransformer(config=model_params)
    model_dict = torch.load(args.model_path)
    # print(model_dict)
    model.load_state_dict(model_dict['model'])
    model = model.to(device)
    model.eval()

    # test(model, "Il y a plusieurs années, ici à Ted, Peter Skillman a présenté une épreuve de conception appelée l'épreuve du marshmallow.", 'Several years ago here at Ted, Peter Skillman presented a design test called the marshmallow test.', wrapped_tokenizer, device=device)
    # exit()

    # 输入文本
    src_text = args.src_text

    # greaedy search
    print('=== greedy search ===')
    greedy = greedy_search(model, src_text, wrapped_tokenizer, max_len=model_params['seq_len'], device=device)

    # beam search
    print('=== beam search ===')
    beam = beam_search(model, src_text, wrapped_tokenizer, max_len=model_params['seq_len'], beam_size=5, device=device)