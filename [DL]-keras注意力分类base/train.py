import tensorflow as tf

attention_keras = __import__("layers")
tdrop = __import__("targetedDropout")

#########################################################################################################
#定义参数
num_words = 20000
maxlen = 80
batch_size = 32

#加载数据
print('Loading data...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(path='./imdb.npz', num_words=num_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
# print(x_train[0])
# print(y_train[:10])

word_index = tf.keras.datasets.imdb.get_word_index('./imdb_word_index.json')
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])

decoded_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in x_train[0]])
# print(decoded_newswire)

#数据对齐
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# print('Pad sequences x_train shape:', x_train.shape)

#########################################################################################################
##### Model ######
S_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32')

embeddings = tf.keras.layers.Embedding(num_words, 128)(S_inputs)
embeddings = attention_keras.Position_Embedding()(embeddings)  #默认使用同等维度的位置向量

attnout = attention_keras.Attention(8, 16)([embeddings, embeddings, embeddings])
print("attnout", attnout)

attnout = tf.keras.layers.GlobalAveragePooling1D()(attnout)

#attnout = tf.keras.layers.Dropout(0.5)(attnout)
attnout = tdrop.TargetedDropout(drop_rate=0.5, target_rate=0.5)(attnout)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(attnout)
print(outputs)

model = tf.keras.models.Model(inputs=S_inputs, outputs=outputs)

#########################################################################################################
##### Train #####
#添加反向传播节点
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#开始训练
print('Train...')
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
