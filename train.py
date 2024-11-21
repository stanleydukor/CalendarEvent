import os
import wandb
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from trainer import TrainingModule
from utils import *

config = load_config('config.yaml')
checkpoint_dir = f'checkpoints'
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss_epoch',
    dirpath=checkpoint_dir,
    filename='best-checkpoint-{epoch}-{val_loss:.2f}',
    save_top_k=3,
    save_last=True,
    mode='min',
)

last_checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
if os.path.exists(last_checkpoint_path):
    resume_from_checkpoint = last_checkpoint_path
else:
    resume_from_checkpoint = None

if __name__ == '__main__':
    fix_random_seed(32)
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_float32_matmul_precision('high' or 'highest')

    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb_logger = WandbLogger(project='Calendar Event Detection', name="model-{}".format(current_timestamp()))
    wandb_logger.experiment.config['batch_size'] = config['batch_size']

    model = TrainingModule(config)
    wandb.watch(model, log_freq=100)
    trainer = Trainer(
        accelerator='gpu',
        devices=[config['device']],
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=resume_from_checkpoint
    )
    trainer.fit(model)
    trainer.test(model)

    wandb.finish()