import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
import timm
import nni

from config import CFG
from dataloader import prepare_dataloader
from utils import seed_everything
from loss import bi_tempered_logistic_loss, MyCrossEntropyLoss


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''
    def forward(self, x):
        x = self.model(x)
        return x


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False):
    model.train()

    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        # print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)
            # print(image_preds.shape, exam_pred.shape)

            loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'

                pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()
    return (image_preds_all == image_targets_all).mean()


if __name__ == '__main__':
    seed_everything(CFG['seed'])

    train_csv_path = r'../data/train.csv' #数据集train.csv路径
    train_img_path = r'../daat/train_images/' #训练集图片路径
    fold_num = 0 # first is 0，每次输入不同的  fold_num  进行训练。总共训练 CFG['fold_num'] 次。

    train = pd.read_csv(train_csv_path)
    train.head()
    train.label.value_counts()

    try:
        tuner_params = nni.get_next_parameter()
        optimizer_type = tuner_params['optimizer']

        folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(
            np.arange(train.shape[0]), train.label.values)

        # we'll train fold 0 first
        # 每次输入不同的  fold_num  进行训练。总共训练 CFG['fold_num'] 次。
        for fold, (trn_idx, val_idx) in enumerate(folds):

            if fold == fold_num:
                print('Training with {} started'.format(fold))

                # print(len(trn_idx), len(val_idx))
                train_loader, val_loader = prepare_dataloader(tuner_params, train, trn_idx, val_idx, data_root=train_img_path)
                device = torch.device(CFG['device'])

                model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)
                scaler = GradScaler()
                if optimizer_type =='Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
                if optimizer_type == 'Adamax':
                    optimizer = torch.optim.Adamax(model.parameters(),lr=CFG['lr'],betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
                if optimizer_type == 'RMSprop':
                    optimizer = torch.optim.RMSprop(model.parameters(),lr=CFG['lr'],alpha=0.99, eps=1e-08, weight_decay=CFG['weight_decay'], momentum=0, centered=False)
                if optimizer_type == 'Adagrad':
                    optimizer = torch.optim.Adagrad(model.parameters(), lr=CFG['lr'], lr_decay=0, weight_decay=CFG['weight_decay'])

                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
                # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
                #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

                loss_tr = nn.CrossEntropyLoss().to(device)
                loss_fn = nn.CrossEntropyLoss().to(device)

                # loss_tr = MyCrossEntropyLoss().to(device)
                # loss_fn = MyCrossEntropyLoss().to(device)

                # loss_tr = lambda y_pred, y_true: bi_tempered_logistic_loss(y_pred, y_true, t1=0.5, t2=1.5)
                # loss_fn = lambda y_pred, y_true: bi_tempered_logistic_loss(y_pred, y_true, t1=0.5, t2=1.5)

                for epoch in range(CFG['epochs']):
                    train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scaler, scheduler=scheduler, schd_batch_update=False)

                    with torch.no_grad():
                        valid_acc = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)

                    nni.report_intermediate_result(valid_acc)
                    # torch.save(model.state_dict(), 'result/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))
                nni.report_final_result(valid_acc)
                # torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
                del model, optimizer, train_loader, val_loader, scaler, scheduler
                torch.cuda.empty_cache()

    except Exception as exception:
        print(exception)
        raise