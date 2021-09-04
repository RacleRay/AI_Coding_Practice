import tensorflow as tf
import numpy as np
from model import MKR


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    kg = data[7]  # 电影实体关系

    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)

    # 只选择user_num个数据，进行测试
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    # 评估top k的效果
    k_list = [1, 2, 5, 10, 20, 50, 100]

    item_set = set(list(range(n_item)))

    model = MKR(args, n_user, n_item, n_entity, n_relation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.n_epochs):
            np.random.shuffle(train_data)
            start = 0
            while (start + args.batch_size) <= train_data.shape[0]:
                feed_dict = get_feed_dict_for_rs(model, train_data, start,
                                                 start + args.batch_size)
                _, loss = model.train_rs(sess, feed_dict)
                start += args.batch_size
                if show_loss:
                    print(f"epoch {epoch}, loss: {loss}")

            # 控制kge部分的训练频率
            if epoch % args.kge_interval == 0:
                start = 0
                while (start + args.batch_size) <= kg.shape[0]:
                    feed_dict = get_feed_dict_for_kge(model, kg, start,
                                                      start + args.batch_size)
                    _, rmse = model.train_kge(sess, feed_dict)
                    start += args.batch_size
                    if show_loss:
                        print(rmse)

            train_auc, train_acc = model.eval(
                sess,
                get_feed_dict_for_rs(model, train_data, 0, train_data.shape[0]))
            eval_auc, eval_acc = model.eval(
                sess,
                get_feed_dict_for_rs(model, eval_data, 0, eval_data.shape[0]))
            test_auc, test_acc = model.eval(
                sess,
                get_feed_dict_for_rs(model, test_data, 0, test_data.shape[0]))
            print(
                '\nepoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                % (epoch, train_auc, train_acc, eval_auc, eval_acc, test_auc,
                   test_acc))

            if show_topk:
                precision, recall, f1 = topk_eval(sess, model, user_list,
                                                  train_record, test_record,
                                                  item_set, k_list)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f' % i, end=' ')
                print('\nrecall:'.ljust(len('precision: ') + 1), end='')
                for i in recall:
                    print('%.4f' % i, end=' ')
                print('\nf1:'.ljust(len('precision: ') + 1), end='')
                for i in f1:
                    print('%.4f' % i, end=' ')


def get_user_record(data, is_train):
    "train阶段，返回用户的所有交互数据，喜欢或者不喜欢；非train，只返回用户喜欢的数据"
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def get_feed_dict_for_rs(model, data, start, end):
    feed_dict = {
        model.user_indices: data[start:end, 0],
        model.item_indices: data[start:end, 1],
        model.labels: data[start:end, 2],
        model.head_indices: data[start:end, 1],
        # 占位，没有实际意义
        model.relation_indices: np.array([-1 for _ in range(end - start)]),
        model.tail_indices: np.array([-1 for _ in range(end - start)])
    }
    return feed_dict


def get_feed_dict_for_kge(model, kg, start, end):
    feed_dict = {
        model.item_indices: kg[start:end, 0],
        model.head_indices: kg[start:end, 0],
        model.relation_indices: kg[start:end, 1],
        model.tail_indices: kg[start:end, 2]
    }
    return feed_dict


def topk_eval(sess, model, user_list, train_record, test_record, item_set,
              k_list):
    "user_list: 选取的测试用户；  item_set：所有电影集合；    k_list：top k选取的k值"
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        # 对不存在于train set中的数据进行预测推荐
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        items, scores = model.get_scores(
            sess, {
                model.user_indices: [user] * len(test_item_list),
                model.item_indices: test_item_list,
                model.head_indices: test_item_list
            })
        # 对推荐电影的分数进行排序
        for item, score in zip(items, scores):
            item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(),
                                        key=lambda x: x[1],
                                        reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        # 当推荐个数为[1, 2, 5, 10, 20, 50, 100]时，模型的精确率与召回率
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]  # 精确率的平均值
    recall = [np.mean(recall_list[k]) for k in k_list]  # 召回率的平均值
    f1 = [2 / (1 / precision[i] + 1 / recall[i])
          for i in range(len(k_list))]  # f1 score
    return precision, recall, f1


if __name__ == '__main__':
    np.random.seed(3)
    tf.reset_default_graph()

    import argparse
    from data_loader import load_data

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='movie',
                        help='which dataset to use')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=10,
                        help='the number of epochs')
    parser.add_argument('--dim',
                        type=int,
                        default=32,
                        help='dimension of user and entity embeddings')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='number of low layers')
    parser.add_argument('--num_score_layer',
                        type=int,
                        default=1,
                        help='number of score layers')
    parser.add_argument('--use_inner_product',
                        type=int,
                        default=1,
                        help='use inner product to compute scores, must be 0 or 1')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='batch size')
    parser.add_argument('--l2_weight',
                        type=float,
                        default=1e-6,
                        help='weight of l2 regularization')
    parser.add_argument('--lr_rs',
                        type=float,
                        default=0.03,
                        help='learning rate of RS task')
    parser.add_argument('--lr_kge',
                        type=float,
                        default=0.01,
                        help='learning rate of KGE task')
    parser.add_argument('--kge_interval',
                        type=int,
                        default=2,
                        help='training interval of KGE task')

    show_loss = False
    show_topk = True

    # show_loss = False
    # show_topk = False

    args = parser.parse_args()
    data = load_data(args)
    train(args, data, show_loss, show_topk)
