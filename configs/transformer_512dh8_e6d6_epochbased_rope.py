run_name = 'transformer_512dh8_e6d6_epochbased_rope'

model_params = {
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'src_vocab_size': None,
    'tgt_vocab_size': None,
    'seq_len': 256,

    # pos embedding
    'abs_pos': 'none', # 'learnable', 'sinusoidal', 'none'
    'qkv_pos': 'rope' # 'rope',' none'
}

train_params = {
    'batch_size': 56,
    'num_epochs': 20,
    'learning_rate': 0.0003,

    'adam_beta1': 0.9,
    'adam_beta2': 0.98,
    'adam_epsilon': 10e-9,

    'scheduler': 'epochbased',
}

valid_params = {
    'batch_size': 64,
    'val_freq': 1, # 每多少个epoch验证一次
}