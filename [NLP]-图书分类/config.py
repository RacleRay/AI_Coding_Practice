import os
import torch

root_path = os.path.abspath(os.path.dirname(__file__))
log_path = os.path.join(root_path, 'logs', 'logs.log')

train_path = os.path.join(root_path, 'data', 'train.tsv')
dev_path = os.path.join(root_path, 'data', 'dev.tsv')
test_path = os.path.join(root_path, 'data', 'test.tsv')

stop_words_path = os.path.join(root_path, 'data', 'stopwords.txt')
labels_path = os.path.join(root_path, 'data', 'label2id.json')

tfidf_path = os.path.join(root_path, 'save', 'embedding', 'tfidf.bin')
word2vec_path = os.path.join(root_path, 'save', 'embedding', 'w2v.bin')
fasttext_path = os.path.join(root_path, 'save', 'embedding', 'fasttext.bin')
lda_path = os.path.join(root_path, 'save', 'embedding', 'lda.bin')

bert_path = os.path.join(root_path, 'save', 'bert\\')

lgb_path = os.path.join(root_path, 'save', 'lgb\\')


is_cuda = False
device = torch.device('cuda') if is_cuda else torch.device('cpu')


class_list = [x.strip() for x in open(labels_path, encoding='utf-8').readlines()]
num_classes = len(class_list)


if __name__ == "__main__":
    print(bert_path)