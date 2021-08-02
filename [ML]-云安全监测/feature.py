import pandas as pd
import csv
import warnings

warnings.filterwarnings('ignore')


# 读txt文件
def read_data(path):
    content = []
    file = open(path, 'r', encoding='UTF-8')
    for line in file:
        content.append(line.strip())
    file.close()
    return content


# 写txt文件(追加)
def write_data(path, data):
    fos = open(path, "a", encoding="utf-8")
    fos.write(data + '\n')
    fos.close()


# 读csv文件
def read_data_csv(path):
    birth_data = []
    with open(path, encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            birth_data.append(row)
    return birth_data


# 写入csv文件
def write_data_csv(path, data):
    with open(path, "a", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        # writer.writerow(["content_id", "train_subject", "sentiment_value", "sentiment_word"])
        # 写入多行用writerows
        temp = data.split(',')
        # writer.writerows([[temp[0], temp[1], temp[2], temp[3]]])

        temp_list = []
        for i in range(len(temp)):
            temp_list.append(temp[i])
        writer.writerows([temp_list])


# 加入特征
def add_feature():
    train_df = pd.read_csv("../Data/security_test.csv")  # file_id,label,api,tid,index
    train_df['return_value'] = 0
    train_df.to_csv("../Data/security_test2.csv", index=False, sep=',')

    test_df = pd.read_csv("../Data/security_test.csv")  # file_id,label,api,tid,index
    test_df['return_value'] = 0
    test_df.to_csv("../Data/security_test2.csv", index=False, sep=',')


# 数据分析
def data_analysis():
    # data_df=pd.read_csv("train.csv",nrows=100000000)
    data_df = pd.read_csv("Data/security_train.csv")
    data_df.head()
    # print(data_df.head())
    label_df = data_df.groupby(['file_id', 'label'])['label'].unique()
    label_class = label_df.value_counts()
    print(label_class)
    dict = {0: "Normal", 1: "Extortion Virus", 2: "Mining Program", 3: "DDoS Trojan Horse", 4: "Worm Virus",
            5: "Infectious Virus", 6: "Backdoor Program", 7: "Trojan Horse Program"}


# 查看数据
def check_csv():
    data_test_df = pd.read_csv("Data/security_test.csv")  # file_id,label,api,tid,index
    data_train_df = pd.read_csv("Data/security_train.csv")  # file_id,label,api,tid,index
    print(data_test_df.shape, data_train_df.shape)


# add_feature()
# data_analysis()
# check_csv()

################################################################################
#

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import gc
import warnings

warnings.filterwarnings('ignore')


# 对训练集的数据预处理
def train_data_preprocess():
    # 数据读取
    print('正在读取训练集')
    train = pd.read_csv('../Data/security_train.csv')
    train['return_value'] = 0  

    print('正在提取训练集特征')
    # 特征工程 & 验证结果(1-Gram)
    train_data = train[['file_id', 'label']].drop_duplicates()
    train_data.head()
    train_data['label'].value_counts()

    # 提取特征一：api
    feature_list1 = ['count', 'nunique']
    train_data = feature_extraction1(train, train_data, feature_list1)

    # 提取特征二：tid
    feature_list2 = ['count', 'nunique', 'max', 'min', 'median', 'std']
    train_data = feature_extraction2(train, train_data, feature_list2)

    # 提取特征三：index
    feature_list3 = ['count', 'nunique', 'max', 'min', 'median', 'std']
    train_data = feature_extraction3(train, train_data, feature_list3)

    # 训练特征
    train_features = [col for col in train_data.columns if col != 'label' and col != 'file_id']
    train_label = 'label'
    train_X, test_X, train_Y, test_Y = train_test_split(train_data[train_features], train_data[train_label].values,
                                                        test_size=0.33)
    gc.collect()

    # 特征扩充：单个特征间的组合
    train, train_data_, combination_feature = feature_extension(train, train_data)

    # 多特征提取
    train, train_data_, combination_feature = feature_extension_multi(train, train_data_)

    # 采用lgb训练
    lgb_train(train_X, test_X, train_Y, test_Y, train, train_data_, train_features, combination_feature)


# 对测试集的数据预处理
def test_data_preprocess():
    # 生成test数据集特征
    # 数据读取
    print('正在读取测试集')
    test = pd.read_csv('../Data/security_test.csv', nrows=1000)
    test['return_value'] = 0  # 定义一个特征变量
    print(test.shape)

    print('正在提取训练集特征')
    # 特征工程
    test_data = test[['file_id']].drop_duplicates()
    test_data.head()

    # 提取特征一：api
    feature_list1 = ['count', 'nunique']
    test_data = feature_extraction1(test, test_data, feature_list1)

    # 提取特征二：title_id
    feature_list2 = ['count', 'nunique', 'max', 'min', 'median', 'std']
    test_data = feature_extraction2(test, test_data, feature_list2)

    # 提取特征三：index
    feature_list3 = ['count', 'nunique', 'max', 'min', 'median', 'std']
    test_data = feature_extraction3(test, test_data, feature_list3)

    # 训练特征 & 标签
    test, test_data_, combination_feature = feature_extension_test(test, test_data)
    test_data, test_data_, combination_feature = feature_extension_multi_test(test, test_data, test_data_)
    test_data_.to_csv('Data/feature/test_data.csv', index=None)


def feature_combination(data_merge, data_orig, combination_feature, col1=None, col2=None, opts=None):
    for opt in opts:
        # print(opt)
        train_split = data_orig.groupby(['file_id', col1])[col2].agg(
            {'fileid_' + col1 + '_' + col2 + '_' + str(opt): opt}).reset_index()

        train_split_ = pd.pivot_table(train_split, values='fileid_' + col1 + '_' + col2 + '_' + str(opt),
                                      index=['file_id'], columns=[col1])
        new_cols = ['fileid_' + col1 + '_' + col2 + '_' + opt + '_' + str(col) for col in train_split_.columns]

        combination_feature.append(new_cols)
        train_split_.columns = new_cols

        train_split_.reset_index(inplace=True)

        data_merge = pd.merge(data_merge, train_split_, how='left', on='file_id')
    return data_merge, combination_feature