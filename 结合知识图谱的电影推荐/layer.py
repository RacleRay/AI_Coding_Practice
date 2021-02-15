import tensorflow as tf
from tensorflow.python.layers import base


class CrossCompressLayer(base.Layer):
    def __init__(self, dim, name=""):
        super(CrossCompressLayer, self).__init__(name)
        self.dim = dim
        self.f_vv = tf.layers.Dense(1, use_bias=False)
        self.f_ev = tf.layers.Dense(1, use_bias=False)
        self.f_ve = tf.layers.Dense(1, use_bias=False)
        self.f_ee = tf.layers.Dense(1, use_bias=False)
        self.bias_v = self.add_weight(name='bias_v',
                                      shape=dim,
                                      initializer=tf.zeros_initializer())
        self.bias_e = self.add_weight(name='bias_e',
                                      shape=dim,
                                      initializer=tf.zeros_initializer())

    def _call(self, inputs):
        "return: 交叉融合后的特征v_out和e_out，分别输入 预测评分部分 和 图谱嵌入部分"
        # v 表示来自 预测评分部分 的中间层向量
        # e 表示来自 图谱嵌入部分 的中间层向量
        v, e = inputs
        v = tf.expand_dims(v, dim=2)
        e = tf.expand_dims(e, dim=1)

        cross = tf.matmul(v, e)
        cross_trans = tf.transpose(cross, perm=[0, 2, 1])
        cross = tf.reshape(cross, [-1, self.dim])
        cross_trans = tf.reshape(cross_trans, [-1, self.dim])

        v_out = tf.reshape(
            self.f_vv(cross) + self.f_ev(cross_trans),
            [-1, self.dim]) + self.bias_v
        e_out = tf.reshape(
            self.f_ve(cross) + self.f_ee(cross_trans),
            [-1, self.dim]) + self.bias_e

        return v_out, e_out
