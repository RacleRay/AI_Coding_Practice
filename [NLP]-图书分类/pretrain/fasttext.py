import pandas as pd
from tqdm import tqdm
import fasttext
from gensim import models

import config
from utils import set_logger

logger = set_logger(config.root_path + '\logs\\fasttext.log')


def train_fasttext(data):
    "使用gensim"
    model = models.FastText(
        data,
        size=300,
        window=3,
        alpha=0.03,
        min_count=2,
        iter=30,
        max_n=3,
        word_ngrams=2,
        max_vocab_size=50000)
    model.wv.save_word2vec_format(config.fasttext_path)


def load_fasttext(path):
    return models.KeyedVectors.load_word2vec_format(
                config.fasttext_path, binary=False)


class Fasttext(object):
    """
    使用facebook开源包，而不是gensim
    """
    def __init__(self,
                 train_raw_path,
                 test_raw_path,
                 model_train_file,
                 model_test_file,
                 model_path=None):
        if model_path is None:
            self.train_raw_data = pd.read_csv(train_raw_path, '\t')
            self.test_raw_data = pd.read_csv(test_raw_path, '\t')
            self.data_process(self.train_raw_data, model_train_file)
            self.data_process(self.test_raw_data, model_test_file)
            self.train(model_train_file, model_test_file)
        else:
            self.fast = fasttext.load_model(model_path)

    def data_process(self, raw_data, model_data_file):
        '''
        处理成专用格式
        '''
        with open(model_data_file, 'w') as f:
            for index, row in tqdm(raw_data.iterrows(),
                                   total=raw_data.shape[0]):
                outline = row['text'] + "\t__label__" + str(
                    int(row['category_id'])) + "\n"
                f.write(outline)

    def train(self, model_train_file, model_test_file):
        self.classifier = fasttext.train_supervised(model_train_file,
                                                    label="__label__",
                                                    dim=50,
                                                    epoch=5,
                                                    lr=0.1,
                                                    wordNgrams=2,
                                                    loss='softmax',
                                                    thread=50,
                                                    verbose=True)
        self.test(model_train_file, model_test_file)
        self.classifier.save_model(config.root_path +
                                   '\model\\fasttext.model', )

    def test(self, model_train_file, model_test_file):
        test_result = self.classifier.test(model_test_file)
        train_result = self.classifier.test(model_train_file)

        # 分别为 精确率和召回率
        print(test_result[1], test_result[2])
        print(train_result[1], train_result[2])


if __name__ == "__main__":
    content = Fasttext()