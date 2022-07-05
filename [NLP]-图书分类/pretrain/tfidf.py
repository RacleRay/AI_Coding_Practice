import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def train_tfidf(data, save_path, stop_words=None):
    count_vect = TfidfVectorizer(stop_words=stop_words,
                                 max_df=0.4,
                                 min_df=0.001,
                                 ngram_range=(1, 2, 3))

    tfidf = count_vect.fit(data)

    joblib.dump(tfidf, save_path)


def load_tfidf(path):
    model = joblib.load(path)
    return model