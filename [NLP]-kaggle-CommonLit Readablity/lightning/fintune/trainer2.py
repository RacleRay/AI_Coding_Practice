import gc
import math
import numpy as np
import pandas as pd

from collections import OrderedDict

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoConfig
from transformers import (get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

from pooler import WeightedLayerPooling
from config import CONFIG


# def set_seed(seed=42):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)


# set_seed(CONFIG.seed)



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.config = AutoConfig.from_pretrained(CONFIG.model)
        self.roberta = AutoModel.from_pretrained(CONFIG.model,
                                                num_labels=1,
                                                output_attentions=False,
                                                output_hidden_states=True)
        self.layer_norm = nn.LayerNorm(CONFIG.hidden_size, eps=1e-5)
        self.pooler = WeightedLayerPooling(CONFIG.num_hidden_layers,
                                           CONFIG.layer_start)
        self.dropout = nn.Dropout(p=0.5)
        self.regressor = nn.Linear(CONFIG.hidden_size, CONFIG.num_labels)

        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[2]

        pool_out = self.pooler(all_hidden_states)[:, 0]
        pool_out = self.layer_norm(pool_out)

        if CONFIG.use_multi_sample_dropout:
            logits = torch.mean(
                torch.stack([self.regressor(self.dropout(pool_out)) for _ in range(5)], dim=0),
                dim=0,
            )
        else:
            logits = self.regressor(pool_out)

        return logits


class Model(pl.LightningModule):
    def __init__(self, fold=None, num_samples=None):
        super(Model, self).__init__()
        self.model = MyModel()

        if fold is not None:
            self.fold = fold

        if num_samples:
            self.train_data_len = num_samples

        self.all_targets = []
        self.all_preds = []
        self.train_loss = 0
        self.val_loss = 0
        self.t_data_size = 0
        self.v_data_size = 0
        self.automatic_optimization = True

        self.best_val_rmse = 1024

    def configure_optimizers(self):
        bert_params = list(self.model.named_parameters())

        no_decay = ["bias", "gamma", "beta", "LayerNorm.weight"]
        group1=['embeddings', 'layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.']
        group2=['layer.8.','layer.9.','layer.10.','layer.11.','layer.12.','layer.13.','layer.14.','layer.15.']
        group3=['layer.16.','layer.17.','layer.18.','layer.19.','layer.20.','layer.21.','layer.22.','layer.23.']
        group_all=group1 + group2 + group3
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': CONFIG.weight_decay, 'lr': CONFIG.learning_rate * 100},
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': CONFIG.weight_decay, 'lr': CONFIG.learning_rate/2.5},
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': CONFIG.weight_decay, 'lr': CONFIG.learning_rate},
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': CONFIG.weight_decay, 'lr': CONFIG.learning_rate*2.5},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0, 'lr': CONFIG.learning_rate * 100},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': CONFIG.learning_rate/2.5},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': CONFIG.learning_rate},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': CONFIG.learning_rate*2.5},
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=CONFIG.learning_rate,
        )

        num_update_steps_per_epoch = math.ceil(self.train_data_len / CONFIG.grad_accum)
        max_train_steps = CONFIG.epochs * num_update_steps_per_epoch

        warmup_steps = CONFIG.warmup_steps

        # Defining LR Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps
        )
        self.scheduler = scheduler
        self.optimizer = optimizer
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'val_loss',
                'interval': 'step'
            }
        }

    @staticmethod
    def _loss_fn(output, target):
        return torch.sqrt(nn.MSELoss()(output,target))

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.model(input_ids, attention_mask)

        loss = self._loss_fn(outputs, labels)
        self.train_loss += (loss.item() * len(labels))
        self.t_data_size += len(labels)
        epoch_loss = self.train_loss / self.t_data_size
        self.log('train_loss',
                 epoch_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        tqdm_dict = {'train_loss': loss}
        sch = self.scheduler
        sch.step()
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.model(input_ids, attention_mask)
        loss = self._loss_fn(outputs, labels)

        self.val_loss += (loss.item() * len(labels))
        self.v_data_size += len(labels)
        epoch_loss = self.val_loss / self.v_data_size

        self.all_targets.extend(labels.detach().squeeze(-1).cpu().numpy())
        self.all_preds.extend(outputs.detach().squeeze(-1).cpu().numpy())
        val_rmse = mean_squared_error(self.all_targets,
                                      self.all_preds,
                                      squared=False)

        logs = {
            "val_loss": epoch_loss,
            "val_rmse": val_rmse,
        }
        self.log_dict(logs, on_epoch=True, prog_bar=True, logger=True)
        output = OrderedDict({
            "val_loss": epoch_loss,
            "val_rmse": val_rmse,
        })
        return output

    def validation_epoch_end(self, outputs):
        val_loss = outputs[-1]["val_loss"]
        tqdm_dict = {'val_loss': val_loss}

        if self.best_val_rmse > outputs[-1]["val_rmse"]:
            self.best_val_rmse = outputs[-1]["val_rmse"]
            torch.save(self.model.state_dict(), f"./model_{self.fold}.bin")

        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss
        }
        return result

    def predict_step(self, batch, batch_idx):
        "batch predict"
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        return self.model(input_ids, attention_mask)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))