import os
import argparse

import torch
import heapq

from data.dataset import get_dataloader
from tokenizer.fr_en_tokenizer import Tokenizer

from models.transformer import TranslationTransformer
from utils.helper import MaskMaker, format_shifted_tgt

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Metric
from ignite.contrib.handlers import ProgressBar

from nltk.translate.bleu_score import corpus_bleu

class BLEUScore(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(BLEUScore, self).__init__(output_transform=output_transform)
        self._predictions = []
        self._references = []

    def reset(self):
        self._predictions = []
        self._references = []

    def update(self, output):
        y_pred, y = output
        self._predictions.extend([y_pred])
        self._references.extend([[ref] for ref in y])  # nltk expects a list of reference translations

    def compute(self):
        return corpus_bleu(self._references, self._predictions)

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
        if beam[0][1][0][-1].item() == wrapped_tokenizer.tgt_tokenizer.eos_token_id:
            break
    
    text = wrapped_tokenizer.decode_output(beam[0][1][0])

    return beam[0][1], text

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
    
    text = wrapped_tokenizer.decode_output(input_tgt[0])

    return input_tgt, text
    

def arg_parse():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--model_path', type=str, default='./ckpt_bkp/demo_lr0.00005_acc0.56_loss1.9171.pt', help='model path')
    parser.add_argument('--config', type=str, default='configs/transformer_512dh8_e6d6.py', help='config file')
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

    # test loader
    test_loader = get_dataloader("test", batch_size=1, wrapped_tokenizer=wrapped_tokenizer)

    model_params['src_vocab_size'] = wrapped_tokenizer.src_vocab_size
    model_params['tgt_vocab_size'] = wrapped_tokenizer.tgt_vocab_size
    # model
    mask_maker = MaskMaker(wrapped_tokenizer)
    model = TranslationTransformer(config=model_params)
    model_dict = torch.load(args.model_path)
    # print(model_dict)
    model.load_state_dict(model_dict['model'])
    model = model.to(device)
    model.eval()

    # # 输入文本
    # src_text = args.src_text

    # # greaedy search
    # print('=== greedy search ===')
    # greedy = greedy_search(model, src_text, wrapped_tokenizer, max_len=model_params['seq_len'], device=device)

    # # beam search
    # print('=== beam search ===')
    # beam = beam_search(model, src_text, wrapped_tokenizer, max_len=model_params['seq_len'], beam_size=5, device=device)

    # test bleu on test set

    def evaluate(engine, batch):
        model.eval()
        src_text, tgt_text, *_ = batch
        output_tgt, output_text = greedy_search(model, src_text[0], wrapped_tokenizer, max_len=model_params['seq_len'], device=device)
        return str(output_text), tgt_text
    
    evaluator = Engine(evaluate)
    bleu = BLEUScore()
    bleu.attach(evaluator, 'bleu')
    pbar = ProgressBar(persist=True).attach(evaluator)

    evaluator.run(test_loader)

    print('BLEU:', evaluator.state.metrics['bleu'])

