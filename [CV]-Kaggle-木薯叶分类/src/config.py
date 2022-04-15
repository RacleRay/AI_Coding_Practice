CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b5_ns',
    'img_size': 512,
    'epochs': 15,
    'train_bs': 8,
    'valid_bs': 8,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 0,
    'accum_iter': 2, # batch accumulation for larger batch size
    'verbose_step': 1,
    'device': 'cuda:0'
}