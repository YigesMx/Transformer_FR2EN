# Report

## Commands

### Sinusoidal Positional Encoding

```bash
# training
python train.py --config configs/transformer_512dh8_e6d6_epochbased_sinusoidal.py
# inference
python inference.py --config configs/transformer_512dh8_e6d6_epochbased_sinusoidal.py --model_path runs/transformer_512dh8_e6d6_epochbased_sinusoidal_20240515_224821/checkpoints/best_checkpoint_checkpoint_19_loss=-2.1470.pt --src_text "Ce dont vous avez peur n’est jamais aussi grave que ce que vous imaginez. La peur que vous laissez s'accumuler dans votre esprit est pire que la situation qui existe réellement."
# test
python test.py --config configs/transformer_512dh8_e6d6_epochbased_sinusoidal.py --model_path runs/transformer_512dh8_e6d6_epochbased_sinusoidal_20240515_224821/checkpoints/best_checkpoint_checkpoint_19_loss=-2.1470.pt
```

### Learnable Positional Encoding

```bash
# training
python train.py --config configs/transformer_512dh8_e6d6_epochbased_learnable.py
# inference
python inference.py --config configs/transformer_512dh8_e6d6_epochbased_learnable.py --model_path runs/transformer_512dh8_e6d6_epochbased_learnable_20240516_102957/checkpoints/best_checkpoint_checkpoint_20_loss=-2.2404.pt --src_text "Ce dont vous avez peur n’est jamais aussi grave que ce que vous imaginez. La peur que vous laissez s'accumuler dans votre esprit est pire que la situation qui existe réellement."
# test
python test.py --config configs/transformer_512dh8_e6d6_epochbased_learnable.py --model_path runs/transformer_512dh8_e6d6_epochbased_learnable_20240516_102957/checkpoints/best_checkpoint_checkpoint_20_loss=-2.2404.pt
```

### Rotary Positional Encoding(RoPE)

```bash
# training
python train.py --config configs/transformer_512dh8_e6d6_epochbased_rope.py
# inference
python inference.py --config configs/transformer_512dh8_e6d6_epochbased_rope.py --model_path runs/transformer_512dh8_e6d6_epochbased_rope_20240516_011720/checkpoints/best_checkpoint_checkpoint_19_loss=-1.7915.pt --src_text "Ce dont vous avez peur n’est jamais aussi grave que ce que vous imaginez. La peur que vous laissez s'accumuler dans votre esprit est pire que la situation qui existe réellement."
# test
python test.py --config configs/transformer_512dh8_e6d6_epochbased_rope.py --model_path runs/transformer_512dh8_e6d6_epochbased_rope_20240516_011720/checkpoints/best_checkpoint_checkpoint_19_loss=-1.7915.pt
```

## Results

| Model      | Train Loss   | Validation Loss   | Validation Accuracy   | Test BLEU-4   |
| ---        | ---          | ---               | ---                   | ---           |
| Sinusoidal | 2.151        | 2.419             | 0.5648                | 0.3187        |
| Learnable  | 2.240        | 2.499             | 0.5538                | 0.3717        |
| RoPE       | 1.857        | 2.304             | 0.5764                | 0.5594        |

## Test

src_text: "Ce dont vous avez peur n’est jamais aussi grave que ce que vous imaginez. La peur que vous laissez s'accumuler dans votre esprit est pire que la situation qui existe réellement."

ground_truth: "What you are afraid of is never as bad as what you imagine. The fear you let build up in your mind is worse than the situation that actually exists"

### Sinusoidal Positional Encoding

output_texts:

=== greedy search ===
What you never care about yourself, and fear that's worse than the situation that you really exists in your mind is worse than the situation.

=== beam search ===
4.8646 What you're never care about.
5.1953 And what you're never care about.
12.0893 What you're never is, and fear that you are worse than the situation in your mind.
17.9308 What you're never care about, fear that you's fear in your mind is worse than the very much worse than the situation.
18.3417 What you're never care about, fear that you's fear in your mind is worse than the very much worse than the situation.

### Learnable Positional Encoding

output_texts:

=== greedy search ===
What you never cares you're going to be thinking, and the situation that you're going to be in your mind is worse than the situation.

=== beam search ===
16.8675 And what you never is the fear that you're going to say, the fear you's going to engage in your mind.
16.1903 And what you never is the fear that you're going to say, the fear you's going to engage in your minds.
16.7127 And what you never is the fear that you're going to say, the fear you's going to invest in your mind.
17.9518 And what you never is the fear that you're going to say, the fear you's going to engage in your mind..
21.0737 And what you never is the fear that you're going to say, the fear you's going to engage in your mind. worse than the situation.

### RoPE

output_texts:

=== greedy search ===
What you fear is never as severe as you can imagine, the fear that you let it be part of, the fear you will be freeer in your mind, is worse than the situation that really exists.

=== beam search ===
24.3234 What you fear is never as severe as you can imagine, the fear that you allow to accumulate in your mind is worse than the situation that really exists.
24.2914 What you fear is never as severe as you can imagine, the fear that you allow to accumulate in your mind is worse than the situation that actually exists.
25.0699 What you fear is never as severe as what you can imagine, the fear that you allow to accumulate in your mind is worse than the situation that actually exists.
25.0656 What you fear is never as severe as what you can imagine, the fear that you allow to accumulate in your mind is worse than the situation that really exists.
31.2803 What you fear is never as severe as what you can imagine, the fear that you allow to accumulate in your mind is worse than the situation that actually exists. is worse than the fact that exists.