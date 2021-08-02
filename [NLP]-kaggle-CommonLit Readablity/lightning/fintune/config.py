class CONFIG:
    seed = 42
    max_len = 256
    hidden_size = 1024
    num_labels = 1

    train_batch = 8
    valid_batch = 8
    epochs = 10

    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_steps = 0
    splits = 5

    model = './pretrian/roberta-large'
    tokenizer = 'roberta-large'
    # tokenizer.save_pretrained('./tokenizer')

    grad_accum = 3

    val_check_interval = 50

    num_hidden_layers = 24
    layer_start = 21  # WeightedLayerPooling使用

    use_multi_sample_dropout = False