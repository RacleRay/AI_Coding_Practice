## TF

```python
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
```



```python
def create_model(roberta_model):
  input_layer_id = Input(shape=(max_len,) ,dtype=tf.int32, name = 'input_ids')
  input_layer_mask = Input(shape=(max_len,) ,dtype=tf.int32, name = 'attention_mask')
    
  roberta = roberta_model.roberta(input_ids = input_layer_id , attention_mask = input_layer_mask)[0]
  roberta_output = roberta[:,0,:]
  x= Dropout(0.2)(roberta_output)
  predictions = Dense(1,activation='linear')(x)
    
  model = Model(inputs=[input_layer_id, input_layer_mask] , outputs=predictions)
  model.compile(
      optimizer = Adam(learning_rate=1e-5),
      metrics = RootMeanSquaredError(),
      loss = "mse"
  )
  return model
```



```python
with strategy.scope():
  roberta_model = TFRobertaModel.from_pretrained("roberta-base")
  model = create_model(roberta_model)

model.summary()
```



[notebook1](https://www.kaggle.com/bharadwajvedula/tpu-high-speed-roberta-training/notebook#Abstract)

[notebook2](https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-with-huggingface-and-keras)



