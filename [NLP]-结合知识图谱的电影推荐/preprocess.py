import argparse
import numpy as np


def convert_id_to_idx(input_file):
    """对entity和item进行重新编号排序

    Args:
        input_file (string): item_index2entity_id.txt

    Returns:
        (dict, dict): (item id的重新编号, entity id的重新编号)
    """
    item_id2idx = dict()
    entity_id2idx = dict()
    i = 0
    for line in open(input_file, 'r', encoding='utf-8').readlines():
        item_id = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_id2idx[item_id] = i
        entity_id2idx[satori_id] = i
        i += 1

    return item_id2idx, entity_id2idx


def convert_rating(input_file, output_file, item_id2idx):
    """rating>=5表示用户感兴趣，小于5表示不感兴趣，使用1/0标注

    Args:
        input_file (string): 用户打分文件
        output_file (string): 输出文件，保存 ‘用户 item 标注’ 格式的数据
        item_id2idx (dict): id重新排列后的字典
    """

    item_set = set(item_id2idx.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(input_file, 'r', encoding='utf-8').readlines()[1:]:
        array = line.strip().split('::')
        user_id = int(array[0])
        item_id = array[1]
        rating = float(array[2])

        if item_id not in item_id2idx:
            continue
        item_idx = item_id2idx[item_id]

        if rating >= 5.0:
            if user_id not in user_pos_ratings:
                user_pos_ratings[user_id] = set()
            user_pos_ratings[user_id].add(item_idx)
        else:
            if user_id not in user_neg_ratings:
                user_neg_ratings[user_id] = set()
            user_neg_ratings[user_id].add(item_idx)

    user_cnt = 0
    user_id2idx = dict()  # 重排user id
    writer = open(output_file, 'w', encoding='utf-8')
    for user_id, pos_item_set in user_pos_ratings.items():
        if user_id not in user_id2idx:
            user_id2idx[user_id] = user_cnt
            user_cnt += 1
        user_idx = user_id2idx[user_id]
        # 正样本
        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_idx, item))

        unwatched_set = item_set - pos_item_set
        if user_id in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_id]
        # 负样本
        for item in np.random.choice(list(unwatched_set),
                                     size=len(pos_item_set),
                                     replace=False):
            writer.write('%d\t%d\t0\n' % (user_idx, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg(input_file, output_file, entity_id2idx):
    """处理图谱关系文件为目标格式 (head, relation, tail)
    """
    entity_cnt = len(entity_id2idx)
    relation_cnt = 0
    relation_id2idx = dict()

    writer = open(output_file, 'w', encoding='utf-8')
    file = open(input_file, 'r', encoding='utf-8')

    for line in file:
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2idx:
            continue
        head = entity_id2idx[head_old]

        if tail_old not in entity_id2idx:
            entity_id2idx[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2idx[tail_old]

        if relation_old not in relation_id2idx:
            relation_id2idx[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2idx[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == "__main__":
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        type=str,
                        default='movie',
                        help='which dataset to preprocess')
    args = parser.parse_args()

    item_id2idx, entity_id2idx = convert_id_to_idx('data/' + args.d +
                                                   '/item_index2entity_id.txt')
    convert_rating('data/' + args.d + '/ratings.dat',
                   'data/' + args.d + '/ratings_final.txt', item_id2idx)
    convert_kg('data/' + args.d + '/kg.txt',
               'data/' + args.d + '/kg_final.txt', entity_id2idx)
