# Transformer Translator FR-EN

## Description

This project is a translation model from French to English using the Transformer architecture. The model is trained on the [iwslt2017](https://huggingface.co/datasets/iwslt2017) FR-EN sub dataset.

We implemented the model following the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. And we compared the model with `torch.nn.Transformer` from PyTorch. They gained similar results except that the inference phase of the PyTorch model is faster due to the optimized implementation.

Furthermore, we implemented and compared three different positional encodings: `sinusoidal`, `learnable`, and `RoPE`. More details about the experiments can be found in the `report.md`.

## Setup

There is a 'environment.yml' file in the repository. You can create a new conda environment using the following command:

```bash
conda env create -f environment.yml
```

But the conda yml may not work properly due to the PyTorch version, CUDA version, and other dependencies.

Here is the list of essential packages:

- pytorch == 2.2.2         (You can just install from the official website along with torchvision, torchaudio)
- ignite == 0.5.0.post2
- huggingface-hub >= 0.22.2
- datasets >= 2.19.0       (for iwslt2017 dataset)
- transformers >= 4.40.1   (for tokenizer)
- nltk >= 3.8.1            (for BLEU-4 score)
- tensorboard           (for visualizing logs)

For attn visualization:
- ipykernal
- matplotlib
- seaborn

You can install the other packages with latest version when you need them.


## Usage

To train the model, you can run the following command:

```bash
python train.py --config PATH_TO_CONFIG_FILE.py
tensorboard --logdir runs
```

To translate a sentence, you can run the following command:

```bash
python inference.py --config PATH_TO_CONFIG_FILE.py --model_path PATH_TO_MODEL.pt --src_text "YOUR_SENTENCE"
```

To test the BLEU-4 score on test dataset, you can run the following command:

```bash
python test.py --config PATH_TO_CONFIG_FILE.py --model_path PATH_TO_MODEL.pt
```