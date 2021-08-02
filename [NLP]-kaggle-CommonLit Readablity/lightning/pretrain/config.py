class TrainConfig:
    train_file = 'mlm_data.csv'
    validation_file = 'mlm_data.csv'
    pad_to_max_length = True

    seed = 2021

    model_type = 'roberta'
    model_name_or_path = 'roberta-large'
    tokenizer_name = 'roberta-large'

    use_slow_tokenizer = True
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8

    learning_rate = 2e-5  # 5e-5
    weight_decay = 0.0

    gradient_accumulation_steps = 1
    lr_scheduler_type = 'constant_with_warmup'
    num_train_epochs = 5
    max_train_steps = None
    num_warmup_steps = 0

    max_seq_length = 256
    line_by_line = False
    preprocessing_num_workers = 4
    overwrite_cache = True

    mlm_probability = 0.15

    output_dir = 'output'

