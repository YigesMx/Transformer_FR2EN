# Transformer Translator FR-EN

## Description

This project is a translation model from French to English using the Transformer architecture. The model is trained on the [iwslt2017](https://huggingface.co/datasets/iwslt2017) FR-EN sub dataset.

We implemented the model following the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. And we compared the model with `torch.nn.Transformer` from PyTorch. They gained similar results except that the inference phase of the PyTorch model is faster due to the optimized implementation.

Furthermore, we implemented and compared three different positional encodings: `sinusoidal`, `learnable`, and `RoPE`. More details about the experiments can be found in the `report.md`.

## Requirements

TODO

## Usage

To train the model, you can run the following command:

```bash
python train.py --config PATH_TO_CONFIG_FILE.py
```

To translate a sentence, you can run the following command:

```bash
python inference.py --config PATH_TO_CONFIG_FILE.py --model_path PATH_TO_MODEL.pt --src_text "YOUR_SENTENCE"
```

To test the BLEU-4 score on test dataset, you can run the following command:

```bash
python test.py --config PATH_TO_CONFIG_FILE.py --model_path PATH_TO_MODEL.pt
```