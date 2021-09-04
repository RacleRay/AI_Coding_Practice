import numpy as np
import tensorflow as tf
from itertools import chain
from sklearn.metrics import roc_auc_score
from layer import CrossCompressLayer
"""
将电影关系和历史打分作为输入，预测用户对任何电影的喜好分数。

模型每一层将 电影关系向量 和 用户打分信息向量，特征融合后再输入下一层。

预测评分部分，输入为 用户向量 和 电影向量
图谱嵌入部分，输入为 head电影向量 和 关系向量，预测tail电影向量
交叉融合部分，将 预测评分部分 和 图谱嵌入部分 的中间特征作为输入，输出交叉后的特征
"""


class MKR:
    def __init__(self, args, n_users, n_items, n_entities, n_relations):
        """
        args: 有属性 dim, num_layers, num_score_layer, use_inner_product, l2_weight
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.n_entities = n_entities

        self.vars_rs = []
        self.vars_kge = []

        self._build_inputs()
        self._build_low_layers(args)
        self._build_high_layers(args)
        self._build_loss(args)
        self._build_train(args)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(tf.int32, [None], 'user_indices')
        self.item_indices = tf.placeholder(tf.int32, [None], 'item_indices')
        self.labels = tf.placeholder(tf.float32, [None], 'labels')
        self.head_indices = tf.placeholder(tf.int32, [None], 'head_indices')
        self.tail_indices = tf.placeholder(tf.int32, [None], 'tail_indices')
        self.relation_indices = tf.placeholder(tf.int32, [None],
                                               'relation_indices')

    def _build_low_layers(self, args):
        # embeddings
        self.user_mat = tf.get_variable('user_embedding_matrix',
                                        [self.n_users, args.dim])
        self.item_mat = tf.get_variable('item_embedding_matrix',
                                        [self.n_items, args.dim])
        self.entity_mat = tf.get_variable('entity_embedding_mat',
                                          [self.n_entities, args.dim])
        self.relation_mat = tf.get_variable('relation_embedding_mat',
                                            [self.n_relations, args.dim])

        self.user_embed = tf.nn.embedding_lookup(self.user_mat,
                                                 self.user_indices)
        self.item_embed = tf.nn.embedding_lookup(self.item_mat,
                                                 self.item_indices)
        self.head_embed = tf.nn.embedding_lookup(self.entity_mat,
                                                 self.head_indices)
        self.relation_embed = tf.nn.embedding_lookup(self.relation_mat,
                                                     self.relation_indices)
        self.tail_embed = tf.nn.embedding_lookup(self.entity_mat,
                                                 self.tail_indices)

        # MKR
        for i in range(args.num_layers):
            user_fc = tf.layers.Dense(args.dim, activation=tf.nn.relu)
            tail_fc = tf.layers.Dense(args.dim, activation=tf.nn.relu)
            cross_layer = CrossCompressLayer(args.dim, name=f"cross layer-{i}")
            # network
            self.user_embed = user_fc(self.user_embed)
            self.tail_embed = tail_fc(self.tail_embed)
            self.item_embed, self.head_embed = cross_layer(
                [self.item_embed, self.head_embed])

            self.vars_rs.append(user_fc.variables)
            self.vars_rs.append(cross_layer.variables)
            self.vars_kge.append(tail_fc.variables)
            self.vars_kge.append(cross_layer.variables)

    def _build_high_layers(self, args):
        "计算输出结果"
        # 预测评分部分
        if args.use_inner_product:
            self.scores_rs = tf.reduce_sum(self.user_embed * self.item_embed,
                                           axis=1)
        else:
            self.user_item_concat = tf.concat(
                [self.user_embed, self.item_embed], axis=1)
            for _ in range(args.num_score_layer - 1):
                rs_fc = tf.layers.Dense(args.dim * 2, activation=tf.nn.relu)
                self.user_item_concat = rs_fc(self.user_item_concat)
                self.vars_rs.append(rs_fc.variables)
            out_layer = tf.layers.Dense(1)
            self.scores_rs = tf.squeeze(out_layer(self.user_item_concat))
            self.vars_rs.append(out_layer.variables)
        self.scores_sigmoid = tf.nn.sigmoid(self.scores_rs)

        # 图谱嵌入部分
        self.head_relation_concat = tf.concat(
            [self.head_embed, self.relation_embed], axis=1)
        for _ in range(args.num_score_layer - 1):
            kge_fc = tf.layers.Dense(args.dim * 2, activation=tf.nn.relu)
            self.head_relation_concat = kge_fc(self.head_relation_concat)
            self.vars_kge.append(kge_fc.variables)

        kge_out = tf.layers.Dense(args.dim)
        self.tail_pred = kge_out(self.head_relation_concat)
        self.tail_pred = tf.nn.sigmoid(self.tail_pred)
        self.vars_kge.append(kge_out.variables)

        self.scores_kge = tf.nn.sigmoid(
            tf.reduce_sum(self.tail_embed * self.tail_pred, axis=1))
        self.rmse = tf.reduce_sum(
            tf.sqrt(
                tf.reduce_mean(tf.square(self.tail_embed - self.tail_pred),
                               axis=1)))

    def _build_loss(self, args):
        # 预测评分部分
        self.base_loss_rs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                    logits=self.scores_rs))
        self.l2_loss_rs = tf.nn.l2_loss(self.user_embed) + tf.nn.l2_loss(
            self.item_embed)
        for var in chain.from_iterable(self.vars_rs):
            self.l2_loss_rs += tf.nn.l2_loss(var)
        self.loss_rs = self.base_loss_rs + self.l2_loss_rs * args.l2_weight

        # 图谱嵌入部分
        self.base_loss_kge = -self.scores_kge
        self.l2_loss_kge = tf.nn.l2_loss(self.head_embed) + tf.nn.l2_loss(
            self.tail_embed) + tf.nn.l2_loss(self.relation_embed)
        for var in chain.from_iterable(self.vars_kge):
            self.l2_loss_kge += tf.nn.l2_loss(var)
        self.loss_kge = self.base_loss_kge + self.l2_loss_kge * args.l2_weight

    def _build_train(self, args):
        self.optimizer_rs = tf.train.AdamOptimizer(args.lr_rs).minimize(
            self.loss_rs)
        self.optimizer_kge = tf.train.AdamOptimizer(args.lr_kge).minimize(
            self.loss_kge)

    def train_rs(self, sess, feed_dict):
        "训练预测评分部分"
        return sess.run([self.optimizer_rs, self.loss_rs], feed_dict)

    def train_kge(self, sess, feed_dict):
        "训练图谱嵌入部分"
        return sess.run([self.optimizer_kge, self.loss_kge], feed_dict)

    def eval(self, sess, feed_dict):
        "计算用户对电影的喜好程度"
        labels, scores = sess.run([self.labels, self.scores_sigmoid],
                                  feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if s >= 0.5 else 0 for s in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_sigmoid], feed_dict)
