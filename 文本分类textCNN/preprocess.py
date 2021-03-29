import tensorflow as tf
import preprocessing

positive_data_file = "data/rt-polaritydata/rt-polarity.pos"
negative_data_file = "data/rt-polaritydata/rt-polarity.neg"


def gline(filelist):
    for file in filelist:
        with open(file, "r", encoding='utf-8') as f:
            for line in f:
                yield line


def mydataset(positive_data_file, negative_data_file, batch_size=256):
    filelist = [positive_data_file, negative_data_file]

    x_text = gline(filelist)
    lenlist = [len(x.split(" ")) for x in x_text]
    max_document_length = max(lenlist)

    x_text = gline(filelist)
    vocab_processor = preprocessing.VocabularyProcessor(max_document_length, 5)
    vocab_processor.fit(x_text)
    # example = list(vocab_processor.reverse([list(range(0, len(vocab_processor.vocabulary_)))]))
    # print("id to word：", example)

    def gen():  #循环生成器（不然一次生成器结束就会没有了）
        while True:
            x_text2 = gline(filelist)
            for i, x in enumerate(vocab_processor.transform(x_text2)):
                if i < int(len(lenlist) / 2):
                    onehot = [1, 0]
                else:
                    onehot = [0, 1]
                yield (x, onehot)

    data = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))
    data = data.shuffle(len(lenlist))
    data = data.batch(batch_size).prefetch(1)

    return data, vocab_processor, max_document_length


if __name__ == '__main__':
    # cheak
    data, _, _ = mydataset(positive_data_file, negative_data_file, 64)
    iterator = data.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(80):
            print("batched data 1:", i)
            sess.run(next_element)
