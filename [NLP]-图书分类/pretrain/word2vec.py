from gensim import models


def train_w2v(data, save_path):
    model = models.Word2Vec(min_count=2,
                            window=5,
                            size=300,
                            sg=1,
                            hs=0,
                            sample=6e-5,
                            alpha=0.03,
                            min_alpha=0.0007,
                            negative=15,
                            workers=4,
                            iter=15,
                            max_vocab_size=50000)

    model.build_vocab(data)
    model.train(data,
                total_examples=model.corpus_count,
                epochs=15,
                report_delay=1)

    model.save_word2vec_format(save_path, binary=False)


def load_w2v(path):
    model = models.KeyedVectors.load_word2vec_format(path,
                                                     binary=False)
    return model