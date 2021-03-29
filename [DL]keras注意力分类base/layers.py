import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class Position_Embedding(keras.layers.Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        """
        Args:
            size (int, optional): position embedding的维度，None表示和word维度相同，输入时必须为偶数. Defaults to None.
            mode (str, optional): sum or concat，sum为position embedding和word embedding相加，或者concat. Defaults to 'sum'.
        """
        super(Position_Embedding, self).__init__(**kwargs)
        self.size = size
        self.mode = mode

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])

        position_j = 1. / K.pow(
            10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        # 按照x的1维度累计求和，与arange一样，生成序列。只不过按照x的实际长度来
        position_i = tf.cumsum(K.ones_like(x[:, :, 0]), 1) - 1
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate(
            [K.cos(position_ij), K.sin(position_ij)], 2)

        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(keras.layers.Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head  # 输出的总维度

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(int(input_shape[0][-1]),
                                         self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(int(input_shape[1][-1]),
                                         self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(int(input_shape[2][-1]),
                                         self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def attnmask(self, inputs, seq_len, mode='mul'):
        "inputs: B, H, L, L attention matrix， seq_len： B，1 每个样本的长度"
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)  # cumsum, 后续的mask全变成1
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:  # Q_len, V_len为mask的长度
            Q_seq, K_seq, V_seq, Q_len, V_len = x

        #对Q、K、V做线性变换,一共做nb_head次，每次线性变化成size_per_head维度
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))  # B, H, L, D

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))  # B, H, L, D


        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5  # B, H, L, L
        A = K.permute_dimensions(A, (0, 3, 2, 1))  # B, L, L, H
        A = self.attnmask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))  # B, H, L, L
        A = K.softmax(A)

        #输出并mask
        output = K.batch_dot(A, V_seq, axes=[3, 2])  # B, H, L, D
        output = K.permute_dimensions(output, (0, 2, 1, 3))  # B, L, H, D
        output = K.reshape(output, (-1, K.shape(output)[1], self.output_dim))  # B, L, H*D
        output = self.attnmask(output, Q_len, 'mul')

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)