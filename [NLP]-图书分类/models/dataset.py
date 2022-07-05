import json
import os
import string
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
from transformers import BertModel, BertTokenizer

# import sys
# curPath = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.split(curPath)[0])

import config
from pretrain.tfidf import load_tfidf
from pretrain.word2vec import load_w2v
from pretrain.lda import load_lda, get_lda_features
from utils import set_logger, softmax

logger = set_logger(config.log_path)

ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


class Dataset:
    def __init__(self, debug_mode=False, train_mode=True):
        self.stop_words = open(config.stop_words_path, encoding='utf-8').readlines()

        self.tfidf = load_tfidf(config.tfidata_path)
        self.word2vec = load_w2v(config.word2vec_path)
        self.lda = load_lda(config.lda_path)
        self.bert_tonkenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path).to(config.device)

        self.labelNameToIndex = json.load(open(config.labels_path, encoding='utf-8'))
        self.labelIndexToName = {v: k for k, v in self.labelNameToIndex.items()}

        self.debug_mode = debug_mode
        if train_mode:
            self.__preprocess()

    def __preprocess(self):
        logger.info('load data')
        self.train = pd.read_csv(config.train_path, sep='\t').dropna()
        self.dev = pd.read_csv(config.dev_path, sep='\t').dropna()

        if self.debug_mode:
            self.train = self.train.sample(n=1000).reset_index(drop=True)
            self.dev = self.dev.sample(n=100).reset_index(drop=True)

        # 拼接数据
        self.train["text"] = self.train['title'] + self.train['desc']
        self.dev["text"] = self.dev['title'] + self.dev['desc']

        # 分词
        self.train["queryCut"] = self.train["text"].apply(self.cut)
        self.dev["queryCut"] = self.dev["text"].apply(self.cut)

        # 过滤停止词
        self.train["queryCutRMStopWord"] = self.train["queryCut"].apply(
            lambda x: [word for word in x if word not in self.stop_words])
        self.dev["queryCutRMStopWord"] = self.dev["queryCut"].apply(
            lambda x: [word for word in x if word not in self.stop_words])

        # label 与id的对应关系
        if os.path.exists(config.labels_path):
            labelNameToIndex = json.load(open(config.labels_path, encoding='utf-8'))
        else:
            labelName = self.train['label'].unique()
            labelIndex = list(range(len(labelName)))
            labelNameToIndex = dict(zip(labelName, labelIndex))
            with open(config.labels_path, 'w', encoding='utf-8') as f:
                json.dump({k: v for k, v in labelNameToIndex.items()}, f)

        self.train["labelIndex"] = self.train['label'].map(labelNameToIndex)
        self.dev["labelIndex"] = self.dev['label'].map(labelNameToIndex)

    @staticmethod
    def cut(query):
        return list(jieba.cut(query))

    def process_data(self):
        '''
        method: word2vec, fasttext, tfidata
        return:
            X_train, feature of train set
            X_test, feature of test set
            y_train, label of train set
            y_test, label of test set
        '''
        X_train, y_train = self.__get_feature(self.train)
        X_test, y_test = self.__get_feature(self.dev)
        return X_train, X_test, y_train, y_test

    def __get_feature(self, data):
        data = self.get_embedding_features(data)
        data = self.get_basic_feature(data)
        data = self.get_bert_feature(data)
        data = self.get_lda_feature(data)
        data.fillna(0.0)

        data["labelIndex"] = data["labelIndex"].astype(int)
        cols = [x for x in data.columns if str(x) not in ['labelIndex']]
        X = data[cols]
        y = data['labelIndex']
        return X, y

    def get_embedding_features(self, data):
        logger.info("[...] generate tfidata embedding features ")
        data["queryCutRMStopWords"] = data["queryCutRMStopWord"].apply(lambda x: " ".join(x))

        tfidata_data = pd.DataFrame(
            self.tfidf.transform(data["queryCutRMStopWords"].tolist()).toarray())
        tfidata_data.columns = ['tfidata' + str(i) for i in range(tfidata_data.shape[1])]
        data = pd.concat([data, tfidata_data], axis=1)
        del tfidata_data

        logger.info("[...] generate word2vec embedding features ")
        data['w2v'] = data["queryCutRMStopWord"].apply(
            lambda x: self.__get_wordvecs(x, self.word2vec))
        data['w2v_mean'] = data['w2v'].progress_apply(
            lambda x: np.mean(np.array(x), axis=0))
        data['w2v_max'] = data['w2v'].progress_apply(
            lambda x: np.max(np.array(x), axis=0))

        logger.info("[...] generate window word2vec embedding features ")
        data['w2v_win_2_mean'] = data['w2v'].progress_apply(
            lambda x: self.embedding_within_windows(x, 2, method='mean'))
        data['w2v_win_3_mean'] = data['w2v'].progress_apply(
            lambda x: self.embedding_within_windows(x, 3, method='mean'))
        data['w2v_win_4_mean'] = data['w2v'].progress_apply(
            lambda x: self.embedding_within_windows(x, 4, method='mean'))
        data['w2v_win_2_max'] = data['w2v'].progress_apply(
            lambda x: self.embedding_within_windows(x, 2, method='max'))
        data['w2v_win_3_max'] = data['w2v'].progress_apply(
            lambda x: self.embedding_within_windows(x, 3, method='max'))
        data['w2v_win_4_max'] = data['w2v'].progress_apply(
            lambda x: self.embedding_within_windows(x, 4, method='max'))

        logger.info("[...] generate joint embedding of words and labels ")
        w2v_of_label = np.array([
            self.word2vec.wv.get_vector(self.labelIndexToName[key])
            for key in self.labelIndexToName
                if self.labelIndexToName[key] in self.word2vec.wv.vocab.keys()
        ])
        data['w2v_label_mean'] = data['w2v'].progress_apply(
            lambda x: self.joint_label_embedding(x, w2v_of_label, method='mean'))
        data['w2v_label_max'] = data['w2v'].progress_apply(
            lambda x: self.joint_label_embedding(x, w2v_of_label, method='max'))

        return data

    def __get_autoencoder_feature(self):
        raise NotImplementedError

    def get_basic_feature(self, data):
        def tag_part_of_speech(data):
            # 获取文本的词性， 并计算名词，动词， 形容词的个数
            words = [tuple(x) for x in list(pseg.cut(data))]
            noun_count = len([w for w in words if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
            adjective_count = len([w for w in words if w[1] in ('JJ', 'JJR', 'JJS')])
            verb_count = len([w for w in words if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')])
            return noun_count, adjective_count, verb_count

        def query_cut(query):
            return list(jieba.cut(query))

        logger.info("[...] generate basic nlp features ")
        # 分字
        data['queryCut'] = data['queryCut'].progress_apply(lambda x: [i if i not in ch2en.keys() else ch2en[i] for i in query_cut(x)])
        # 文本的长度
        data['length'] = data['queryCut'].progress_apply(lambda x: len(x))

        # 大写的个数
        data['capitals'] = data['queryCut'].progress_apply(lambda x: sum(1 for c in x if c.isupper()))
        # 大写 与 文本长度的占比
        data['caps_vs_length'] = data.progress_apply(lambda row: float(row['capitals']) / float(row['length']), axis=1)

        # 感叹号的个数
        data['num_exclamation_marks'] = data['queryCut'].progress_apply(lambda x: x.count('!'))
        # 问号个数
        data['num_question_marks'] = data['queryCut'].progress_apply(lambda x: x.count('?'))
        # 标点符号个数
        data['num_punctuation'] = data['queryCut'].progress_apply(lambda x: sum(x.count(w) for w in string.punctuation))
        # *&$%字符的个数
        data['num_symbols'] = data['queryCut'].progress_apply(lambda x: sum(x.count(w) for w in '*&$%'))

        # 词的个数
        data['num_words'] = data['queryCut'].progress_apply(lambda x: len(x))
        # 唯一词的个数
        data['num_unique_words'] = data['queryCut'].progress_apply(lambda x: len(set(w for w in x)))
        # 唯一词 与总词数的比例
        data['words_vs_unique'] = data['num_unique_words'] / data['num_words']

        # 获取名词， 形容词， 动词的个数
        data['nouns'], data['adjectives'], data['verbs'] = zip(*data['text'].progress_apply(lambda x: tag_part_of_speech(x)))
        # 名词占总长度的比率
        data['nouns_vs_length'] = data['nouns'] / data['length']
        # 形容词占总长度的比率
        data['adjectives_vs_length'] = data['adjectives'] / data['length']
        # 动词占总长度的比率
        data['verbs_vs_length'] = data['verbs'] / data['length']
        # 名词占总词数的比率
        data['nouns_vs_words'] = data['nouns'] / data['num_words']
        # 形容词占总词数的比率
        data['adjectives_vs_words'] = data['adjectives'] / data['num_words']
        # 动词占总词数的比率
        data['verbs_vs_words'] = data['verbs'] / data['num_words']

        # 首字母大写其他小写的个数
        data["count_words_title"] = data["queryCut"].progress_apply(lambda x: len([w for w in x if w.istitle()]))

        # 平均词长度
        data["mean_word_len"] = data["text"].progress_apply(lambda x: np.mean([len(w) for w in x]))

        # 标点符号的占比
        data['punct_percent'] = data['num_punctuation'] * 100 / data['num_words']

        return data

    def get_bert_feature(self, data):
        def bertize(text):
            text_dict = self.bert_tonkenizer.encode_plus(
                text,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=400,           # Pad & truncate all sentences.
                ad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids, attention_mask, token_type_ids = text_dict[
                'input_ids'], text_dict['attention_mask'], text_dict['token_type_ids']
            _, res = self.bert(input_ids.to(config.device),
                            attention_mask=attention_mask.to(config.device),
                            oken_type_ids=token_type_ids.to(config.device))
            return res.detach().cpu().numpy()[0]

        logger.info("[...] generate BERT embedding features ")
        data['text'].progress_apply(lambda x: bertize(x))
        return data

    def get_lda_feature(self, data):
        logger.info("[...] generate lda features ")
        # bag of word 格式数据
        data['bow'] = data['queryCutRMStopWord'].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x))

        # 在bag of word 基础上得到文档属于每个topic的概率值的向量，作为一种特征
        data['lda'] = list(
            map(lambda doc: get_lda_features(self.ml_data.em.lda, doc), data['bow']))

        return data

    @staticmethod
    def __wordvec_aggregate(sentence, w2v_model, method='mean'):
        arr = np.array([
            w2v_model.wv.get_vector(s) for s in sentence
            if s in w2v_model.wv.vocab.keys()
        ])

        if len(arr) > 0:
            if method == 'mean':
                return np.mean(np.array(arr), axis=0)
            elif method == 'max':
                return np.max(np.array(arr), axis=0)
            else:
                raise NotImplementedError
        else:
            return np.zeros(300)

    @staticmethod
    def __get_wordvecs(sentence, w2v_model):
        arr = np.array([
            w2v_model.wv.get_vector(s) for s in sentence
            if s in w2v_model.wv.vocab.keys()
        ])
        return arr

    @staticmethod
    def joint_label_embedding(example_matrix, label_embedding, method='mean'):
        '''
        论文 《Joint embedding of words and labels for text classification》
        获取由输入词向量的attention表示的label embedding

        example_matrix: denotes words embedding of input
        label_embedding: denotes the embedding of all label

        return: attention embeddings
        '''
        # [sent len, 300] @ [300, num labels] => [sent len, num labels]
        similarity_matrix = np.dot(example_matrix, label_embedding.T) / (
            np.linalg.norm(example_matrix) * (np.linalg.norm(label_embedding)))

        attention = similarity_matrix.max(axis=1)  # max pooling
        attention = softmax(attention)[:, np.newaxis]

        # [sent len, 300]
        attention_embedding = example_matrix * attention
        if method == 'mean':
            return np.mean(attention_embedding, axis=0)
        else:
            return np.max(attention_embedding, axis=0)

    @staticmethod
    def embedding_within_windows(embedding_matrix, window_size=2, method='mean'):
        result_list = []
        for k1 in range(len(embedding_matrix)):
            if int(k1 + window_size) > len(embedding_matrix):
                result_list.extend(np.mean(embedding_matrix[k1:], axis=0).reshape(1, 300))
            else:
                result_list.extend(
                    np.mean(embedding_matrix[k1:k1 + window_size],axis=0).reshape(1, 300))
        if method == 'mean':
            return np.mean(result_list, axis=0)
        else:
            return np.max(result_list, axis=0)


if __name__ == "__main__":
    a = np.random.normal(size=(10, 300))
    b = np.random.normal(size=(2, 300))
    Dataset.joint_label_embedding(a, b)