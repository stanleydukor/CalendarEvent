import logging
import wandb
import torch
import torch.nn as nn
import lightning.pytorch as ptl
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from Dataset.dataset import ChatDataset
from Model.model import CalendarEventDetector
from utils import *

LOGGER = logging.getLogger(__name__)

class TrainingModule(ptl.LightningModule):
    def __init__(self, config):
        super().__init__()
        LOGGER.info("TrainingModule init called")
        
        self.config = config
        self.loss = nn.BCELoss()
        self.model = CalendarEventDetector(bert_model=config['model'])

        self.dataset = ChatDataset('Data/train.csv')
        self.test_dataset = ChatDataset('Data/test.csv')
        train_size = int(self.config['train_size'] * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        generator = torch.Generator().manual_seed(32)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, valid_size], generator=generator)

        self.validation_step_outputs = []
        self.N_epochs = 5
        self.last_N_losses = []
        self.val_epoch_count = 0

        self.training_table = wandb.Table(columns=["Train Message", "Train True Label", "Train Pred Label"])
        self.validation_table = wandb.Table(columns=["Val Message", "Val True Label", "Val Pred Label"])

        self.messages = []
        self.labels = []
        self.preds = []

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
    
    def configure_optimizers(self):
        optimizer =  optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.config['max_epochs']),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'run_avg_val_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=False)
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=False)
        return dataloader

    def training_step(self, batch, batch_idx):
        self._is_training_step = True
        loss, messages, labels, outputs = self._do_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx % self.config['text_log_step'] == 0:
            pred_label = (outputs[0] > 0.5).float()
            self.training_table.add_data(messages[0], labels[0], pred_label)
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        wandb.log({"train_table": self.training_table})
        self.training_table = wandb.Table(columns=["Train Message", "Train True Label", "Train Pred Label"])

    def validation_step(self, batch, batch_idx):
        self._is_training_step = False
        loss, messages, labels, outputs = self._do_step(batch, batch_idx)
        self.validation_step_outputs.append(loss)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx % self.config['text_log_step'] == 0:
            pred_label = (outputs[0] > 0.5).float()
            self.validation_table.add_data(messages[0], labels[0], pred_label)
        return {"loss": loss}
    
    def on_validation_epoch_end(self):
        avg_epoch_loss = torch.stack(self.validation_step_outputs).mean()
        self.last_N_losses.append(avg_epoch_loss)
        self.last_N_losses = self.last_N_losses[-self.N_epochs:]
        run_avg_val_loss = sum(self.last_N_losses) / len(self.last_N_losses)
        if self.val_epoch_count % self.config['lr_log_step'] == 0:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log(f"lr_per_{self.config['lr_log_step']}_epoch", current_lr, prog_bar=True, logger=True)
        self.log('run_avg_val_loss', run_avg_val_loss, prog_bar=True, logger=True)
        self.val_epoch_count += 1
        self.validation_step_outputs.clear()
        wandb.log({"val_table": self.validation_table})
        self.validation_table = wandb.Table(columns=["Val Message", "Val True Label", "Val Pred Label"])

    def test_step(self, batch, batch_idx):
        self._is_training_step = False
        loss, messages, labels, outputs = self._do_step(batch, batch_idx)
        self.messages.extend(messages)
        self.labels.extend([label.item() for label in labels])
        self.preds.extend([1 if output.item() > 0.5 else 0 for output in outputs])
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def on_test_epoch_end(self):
        table = wandb.Table(columns=["Test Message", "Test True Label", "Test Pred Label"])
        for message, label, pred in zip(self.messages, self.labels, self.preds):
            table.add_data(message, label, pred)
        wandb.log({"test_table": table})

        accuracy, precision, recall, f1 = get_metrics(self.labels, self.preds)
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=self.labels,
                preds=self.preds
            )
        })
    
        self.messages.clear()
        self.labels.clear()
        self.preds.clear()
    
    def _do_step(self, batch, batch_idx):
        messages, labels = batch['message'], batch['label']
        outputs = self.forward(batch['input_ids'], batch['attention_mask']).squeeze()
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        l_bce = self._get_loss(outputs, labels.float())
        return l_bce, messages, labels, outputs
    
    def _get_loss(self, pred, true):
        l_bce = self.loss(pred, true)
        return l_bce