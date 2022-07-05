import numpy as np
import gensim
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
import config


def train_lda(data):
    id2word = gensim.corpora.Dictionary(data.text)
    corpus = [id2word.doc2bow(text) for text in data.text]
    LDAmodel = LdaMulticore(corpus=corpus,
                            id2word=id2word,
                            num_topics=30,
                            workers=4,
                            chunksize=4000,
                            passes=7,
                            alpha='asymmetric')

    LDAmodel.save(config.lda_path)


def load_lda(path):
    return LdaModel.load(path)


def get_lda_features(lda_model, document):
    '''
    Transforms a bag of words document to lda features.
    It returns the proportion of how much each topic was present in the document.

    return: vector of lda topic probabilities of each topic
    '''
    # 基于bag of word 格式数据获取lda的特征
    topic_importances = lda_model.get_document_topics(document,
                                                      minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:, 1]