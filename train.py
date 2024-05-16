import os
import argparse
import time

import torch

from data.dataset import get_dataloader
from tokenizer.fr_en_tokenizer import Tokenizer

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss
from models.transformer import TranslationTransformer
from utils.helper import MaskMaker, format_shifted_tgt

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy, Bleu
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar, TensorboardLogger

# from configs.transformer_512dh8_e6d6 import model_params, train_params, valid_params

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--config', type=str, default='configs/transformer_512dh8_e6d6.py', help='config file')
    return parser.parse_args()

def main(args):
    # config
    from importlib.machinery import SourceFileLoader
    config = SourceFileLoader('config', args.config).load_module()
    model_params = config.model_params
    train_params = config.train_params
    valid_params = config.valid_params
    run_name = config.run_name
    run_dir = f"runs/{run_name}" + time.strftime(r"_%Y%m%d_%H%M%S")
    run_checkpoint_dir = f"{run_dir}/checkpoints"
    run_log_dir = f"{run_dir}/logs"

    from pprint import pprint
    pprint(model_params)
    pprint(train_params)
    pprint(valid_params)

    # device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # tokenizer
    wrapped_tokenizer = Tokenizer(model_params['seq_len'])

    # 加载数据
    train_loader = get_dataloader("train", batch_size=train_params['batch_size'], wrapped_tokenizer=wrapped_tokenizer)
    val_loader = get_dataloader("validation", batch_size=valid_params['batch_size'], wrapped_tokenizer=wrapped_tokenizer)

    # 初始化模型、优化器、损失函数
    model_params['src_vocab_size'] = wrapped_tokenizer.src_vocab_size
    model_params['tgt_vocab_size'] = wrapped_tokenizer.tgt_vocab_size
    mask_maker = MaskMaker(wrapped_tokenizer)

    model = TranslationTransformer(config=model_params).to(device)
    optimizer = Adam(model.parameters(), lr=train_params['learning_rate'], betas=(train_params['adam_beta1'], train_params['adam_beta2']), eps=train_params['adam_epsilon'])
    print(f"lr: {optimizer.param_groups[0]['lr']}")

    if train_params['scheduler'] == 'epochbased':
        print('Using epochbased scheduler')
        def lr_lambda(epoch):
            warmup_epoch = 6
            down = 7
            if epoch < warmup_epoch:
                return (epoch+1) / (warmup_epoch+1)
                # return epoch+1 / warmup_epoch+1
            elif epoch < down:
                return 1
            else: 
                return torch.exp(torch.tensor(-0.35 * (epoch - down))) # 0.1 for more epochs
    elif train_params['scheduler'] == 'iterbased':
        print('Using iterbased scheduler')
        def lr_lambda(iteration): # scheduler used in Attention is All You Need
            warmup_steps = 8000
            step_num = iteration + 1
            arg1 = step_num ** -0.5
            arg2 = step_num * (warmup_steps ** -1.5)
            lr = (model_params['d_model'] ** -0.5) * min(arg1, arg2)
            return lr
    else:
        raise ValueError("scheduler must be 'epochbased' or 'iterbased'")
        
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = CrossEntropyLoss(ignore_index=wrapped_tokenizer.tgt_tokenizer.pad_token_id)
    
    # resume
    if args.resume is not None:
        model.load_state_dict(torch.load("checkpoints/regular_checkpoint_1.pt")['model'])
        optimizer.load_state_dict(torch.load("checkpoints/regular_checkpoint_1.pt")['optimizer'])

    # 接下来创建 trainer 和 evaluator
    # train的每个epoch结束后，会调用 evaluator 进行验证集的验证，也可能单独运行evaluator

    # 创建 trainer
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        # read texts from batch
        src, tgt = batch
        # src: [batch, seq_len]
        input_tgt, output_tgt = format_shifted_tgt(tgt)
        # input_tgt: [batch, seq_len]
        # output_tgt: [batch, seq_len]
        src = src.to(device)
        input_tgt = input_tgt.to(device)
        output_tgt = output_tgt.to(device)

        # create masks
        masks = mask_maker.create_masks(src, input_tgt)

        # forward
        output = model(src=src, tgt=input_tgt, masks=masks)
        # output: [batch, seq_len, tgt_vocab_size]

        # calculate loss
        output_ = output.view(-1, output.shape[-1]) # [batch * seq_len, tgt_vocab_size]
        label_ = output_tgt.view(-1) # [batch * seq_len]
        loss = criterion(output_, label_)

        # backward
        loss.backward()
        if train_params['scheduler'] == 'iterbased':
            lr_scheduler.step()
        optimizer.step()

        return loss.item()
    
    trainer = Engine(train_step)
    
    # 创建 evaluator
    def validation_step(engine, batch):
        model.eval()

        with torch.no_grad():
            src, tgt = batch
            src = src.to(device)
            input_tgt, output_tgt = format_shifted_tgt(tgt)
            input_tgt = input_tgt.to(device)
            output_tgt = output_tgt.to(device)
            masks = mask_maker.create_masks(src, input_tgt)

            output = model(src=src, tgt=input_tgt, masks=masks)
            # -> [batch, seq_len, tgt_vocab_size]

            output_ = output.view(-1, output.shape[-1])
            # -> [batch * seq_len, tgt_vocab_size]
            label_ = output_tgt.view(-1)
            # -> [batch * seq_len]

            # mask pad
            mask = masks['tgt_key_padding_mask'].view(-1)
            output_acc = output_[~mask]
            label_acc = label_[~mask]

            return output_, label_, output_acc, label_acc
    
    # train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    val_metrics = {
        "accuracy": Accuracy(
            output_transform=lambda x: (x[2], x[3])
        ), # Accuracy: 计算准确率
        "loss": Loss(
            criterion,
            output_transform=lambda x: (x[0], x[1])
        )  # Loss: 计算损失
    }
    for name, metric in val_metrics.items():
        # metric.attach(train_evaluator, name)
        metric.attach(val_evaluator, name)

    # 每个epoch结束后，调用train_evaluator进行验证
    # 每model_params['val_freq']个epoch结束后，调用val_evaluator进行验证

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        engine.state.metrics["loss"] = engine.state.output

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(trainer):
    #     train_evaluator.run(train_loader)
    #     metrics = train_evaluator.state.metrics
    #     print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        if trainer.state.epoch % valid_params['val_freq'] == 0:
            val_evaluator.run(val_loader)
            metrics = val_evaluator.state.metrics
            print(f"\nValidation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr(engine):
        if train_params['scheduler'] == 'epochbased':
            lr_scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    # 使用 ProgressBar 显示训练进度
    # 使用 ModelCheckpoint 保存模型
    # 使用 TensorboardLogger 记录训练过程

    # Attach progress bar
    # train bar
    ProgressBar(persist=False).attach(trainer, metric_names="all")
    # val bar
    # ProgressBar(persist=False).attach(train_evaluator, metric_names="all")
    # ProgressBar(persist=False).attach(val_evaluator, metric_names="all")

    # Attach model checkpoint
    regular_checkpoint_handler = ModelCheckpoint(
        dirname=run_checkpoint_dir,
        filename_prefix="regular_checkpoint",
        n_saved=3,
        create_dir=True,
        require_empty=False,
        global_step_transform=lambda *_: trainer.state.epoch,
        # {"model": model, "optimizer": optimizer}
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, regular_checkpoint_handler, {"model": model, "optimizer": optimizer})

    # also after each epoch, check if the loss is the best, if so, save the model
    best_checkpoint_handler = ModelCheckpoint(
        dirname=run_checkpoint_dir,
        filename_prefix="best_checkpoint",
        n_saved=1,
        create_dir=True,
        require_empty=False,
        # save_as_state_dict=True,
        global_step_transform=lambda *_: trainer.state.epoch,
        score_function=lambda engine: -engine.state.metrics["loss"],
        score_name="loss",
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, best_checkpoint_handler, {"model": model, "optimizer": optimizer})

    # Attach tensorboard logger:
    # every epoch, log the train loss and train accuracy
    # every eval, log the val loss and val accuracy
    # every time either checkpoint_handler is called, log the best model
    tb_logger = TensorboardLogger(log_dir=run_log_dir)
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        output_transform=lambda x: x,
    )
    # tb_logger.attach_output_handler(
    #     train_evaluator,
    #     event_name=Events.EPOCH_COMPLETED,
    #     tag="training",
    #     metric_names="all",
    #     global_step_transform=lambda *_: trainer.state.epoch,
    # )
    tb_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names="all",
        global_step_transform=lambda *_: trainer.state.epoch,
    )

    # 开始训练
    trainer.run(train_loader, max_epochs=train_params['num_epochs'])

if __name__ == "__main__":
    # proxy
    # os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    # os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    
    # CUDA_LAUNCH_BLOCKING=1
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    args = parse_args()

    main(args)