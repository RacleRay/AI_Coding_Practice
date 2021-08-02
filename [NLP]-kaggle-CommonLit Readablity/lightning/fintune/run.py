import os
import time
import gc
import numpy as np
import pandas as pd
# from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from transformers import AutoTokenizer

from sklearn.model_selection import StratifiedKFold

from config import CONFIG
from dataset import SmartBatchingDataset, RoBERTaDataset
from trainer import Model

import warnings
warnings.filterwarnings("ignore")


import wandb
from pytorch_lightning.loggers import WandbLogger

wandb.login()


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)

    # initiate the kfold class from model_selection module
    kf = StratifiedKFold(n_splits=num_splits)

    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data


def get_train_data(fold, df):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 使用pl TPU 效率变低，因为pl会使用sampler_ddp优化对TPU的训练
    # train_dataset = SmartBatchingDataset(df_train)
    # tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer)
    # train_loader = train_dataset.get_dataloader(batch_size=CONFIG.train_batch,
    #                                             max_len=CONFIG.max_len,
    #                                             pad_id=tokenizer.pad_token_id)

    train_dataset = RoBERTaDataset(df_train)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.train_batch,
                              num_workers=4, shuffle=False, pin_memory=True)

    valid_dataset = RoBERTaDataset(df_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.valid_batch,
                              num_workers=4, shuffle=False, pin_memory=True)

    return train_loader, valid_loader


def get_test_data(df):
    test_dataset = RoBERTaDataset(df, for_test=True)

    test_loader = DataLoader(test_dataset, batch_size=CONFIG.test_batch,
                             num_workers=4, shuffle=False, pin_memory=True)

    return test_loader


def train(fold, df):
    train_loader, valid_loader = get_train_data(fold, df)

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Checkpoint
    checkpoint_callback = ModelCheckpoint(dirpath=f"./{fold}",
                                          monitor='val_rmse',
                                          save_top_k=1,
                                          save_last=False,
                                          save_weights_only=True,
                                          filename='{epoch:02d}-{val_loss:.4f}-{val_rmse:.3f}',
                                          verbose=False,
                                          mode='min')

    # Earlystopping：TPU使用有问题，没有转换到 TPU tensor，计算报错
    # earlystopping = EarlyStopping(monitor='val_rmse', patience=10, mode='min')

    #########################################################################
    # train
    model = Model(len(train_loader))
    # instrument experiment with W&B
    wandb_logger = WandbLogger(project='CommonlitReadabilityTrain_', log_model='all', job_type='train')

    # 使用自定义 Dataset with custom sampler 时，设置replace_sampler_ddp=False
    # trainer = pl.Trainer(logger=wandb_logger,
    #                     max_epochs=CONFIG.epochs,
    #                     accumulate_grad_batches=CONFIG.grad_accum,
    #                     callbacks=[lr_monitor, checkpoint_callback],
    #                     tpu_cores=1,
    #                     replace_sampler_ddp=False,
    #                     val_check_interval=40)

    trainer = pl.Trainer(logger=wandb_logger,
                         max_epochs=CONFIG.epochs,
                         accumulate_grad_batches=CONFIG.grad_accum,
                         callbacks=[lr_monitor, checkpoint_callback],
                         tpu_cores=1,
                         val_check_interval=40)

    # trainer = pl.Trainer(logger=wandb_logger,
    #                      max_epochs=CONFIG.epochs,
    #                      accumulate_grad_batches=CONFIG.grad_accum,
    #                      callbacks=[lr_monitor, checkpoint_callback],
    #                      gpus=1,
    #                      val_check_interval=40)

    # log gradients and model topology
    wandb_logger.watch(model)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
    wandb.finish()

    print("【Fold {flod}】Saved best_model_path: ", checkpoint_callback.best_model_path)


def test(df, checkpoint_paths):
    test_loader = get_test_data(df)
    results = []
    test_time_list = []

    for path in checkpoint_paths:
        torch.cuda.synchronize()
        tic1 = time.time()

        model = Model()
        model.load_from_checkpoint(path)
        model.eval()
        model.freeze()

        trainer = pl.Trainer(gpus=1)
        predictions = trainer.predict(model, dataloaders=test_loader)
        results.append(predictions)

        torch.cuda.synchronize()
        tic2 = time.time()
        test_time_list.append(tic2 - tic1)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results, test_time_list


def main():
    ################################################################################################
    # Installing PyTorch XLA Frameworks so that we can work with TPU
    #
    # !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
    # !python pytorch-xla-env-setup.py --version 1.7 --apt-packages libomp5 libopenblas-dev
    ################################################################################################

    ################################################################################################
    # # API key ： https://wandb.ai/authorize
    # !pip install --upgrade -q wandb
    # ! wandb login "6b79cd4f90ab6fe735d946f8396628fcc9592f7a"
    # os.system('wandb login "6b79cd4f90ab6fe735d946f8396628fcc9592f7a"')
    ################################################################################################

    set_seed(CONFIG.seed)
    # seed_everything(42)

    ################################################################################################
    # data
    df = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
    test_df = pd.read_csv("../input/commonlitreadabilityprize/test.csv", usecols=["id","excerpt"])
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    def prep_text(df):
        df = df.str.replace("[\n|\t]", " ", regex=True)
        return df.str.replace(" {2,}", " ", regex=True).values
    df["excerpt"] = prep_text(df["excerpt"])
    test_df["excerpt"] = prep_text(test_df["excerpt"])

    df = create_folds(df, num_splits=5)

    for fold in range(5):
        print('=' * 42)
        print(f'[Num FOLD]: {fold} , training...')
        train(fold, df)
        print()