import os
import cv2
import torch
import random
import numpy as np


def seed_everything(seed):
    '''固定各类随机种子，方便消融实验.
    Args:
        seed :  int
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)       # 固定 torch cpu 计算的随机种子
    torch.cuda.manual_seed(seed)  # 固定 cuda 计算的随机种子
    torch.backends.cudnn.deterministic = True  # 是否将卷积算子的计算实现固定。torch 的底层有不同的库来实现卷积算子
    torch.backends.cudnn.benchmark = True  # 是否开启自动优化，选择最快的卷积计算方法


def get_img(path):
    '''使用 opencv 加载图片.
    由于历史原因，opencv 读取的图片格式是 bgr
    Args:
        path : str  图片文件路径 e.g '../data/train_img/1.jpg'
    '''
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def rand_bbox(size, lam):
    '''cutmix 的 bbox 截取函数
    Args:
        size : tuple 图片尺寸 e.g (256,256)
        lam  : float 截取比例，0表示整图全部截取，1表示不截取
    Returns:
        bbox 的左上角和右下角坐标
        int,int,int,int
    '''
    W = size[0]  # 截取图片的宽度
    H = size[1]  # 截取图片的高度
    cut_rat = np.sqrt(1. - lam)  # 需要截取的 bbox 比例
    cut_w = np.int(W * cut_rat)  # 需要截取的 bbox 宽度
    cut_h = np.int(H * cut_rat)  # 需要截取的 bbox 高度

    cx = np.random.randint(W)  # 均匀分布采样，随机选择截取的 bbox 的中心点 x 坐标
    cy = np.random.randint(H)  # 均匀分布采样，随机选择截取的 bbox 的中心点 y 坐标

    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # 左上角 x 坐标
    bby1 = np.clip(cy - cut_h // 2, 0, H)  # 左上角 y 坐标
    bbx2 = np.clip(cx + cut_w // 2, 0, W)  # 右下角 x 坐标
    bby2 = np.clip(cy + cut_h // 2, 0, H)  # 右下角 y 坐标
    return bbx1, bby1, bbx2, bby2