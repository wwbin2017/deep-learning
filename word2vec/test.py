#!/usr/bin/python
# --*-- coding: utf-8 --*--

import utils
import doc2vec
import word2vec


def train_model_word2vec():
    file_name = './data/zh_classicalwiki_extracted_word_seg_result.txt'
    with open(file_name, 'r') as fp:
        words = utils.read_file(fp)
    w2v = word2vec.Word2Vec(batch_size=128,           # 批训练大小
                 skip_window=3,                # 单边窗口长
                 num_skips=2,              #
                 embedding_size=200,       # 词向量的长度
                 vocabulary_size=50000,    # 字典大小
                 num_sampled=64,           # 负采样
                 learning_rate=1e-2,
                 n_steps=100001,            # 训练次数
                 logdir='./model/word2vec/tmp_word2vec')

    w2v.train_model(words)
    w2v.save_model("./model/word2vec")


def load_model_word2vec(path="./model/word2vec"):
    w2v = word2vec.Word2Vec.restore(path)
    word_list = range(100,220)
    test_words, near_words, sim_mean, sim_var, sim_matrix_top = w2v.cal_similarity(word_list)
    for i in  range(len(test_words)):
        print "*"*20
        print test_words[i]
        for j in range(len(near_words[i])):
            print near_words[i][j], "sim", sim_matrix_top[i][j]


def test_word2vec():
    train_model_word2vec()
    print("ok", "word2vec")
    load_model_word2vec()


def train_model_doc2vec():
    docs = utils.docs2list("./data")
    d2v = doc2vec.Doc2Vec(batch_size=256, window_size=8,
                     concat=True,
                     embedding_size_w=64,
                     embedding_size_d=64,
                     vocabulary_size=5000,
                     document_size=len(docs),
                     n_neg_samples=64,
                     learning_rate=1e-5, n_steps=100001)
    d2v.train_model(docs)
    d2v.save("./model/doc2vec")


def load_model_doc2vec(path='./model/doc2vec'):
    d2v = doc2vec.Doc2Vec.restore(path)
    docs = utils.docs2list("./data")
    sim = d2v.predict_doc_predict(len(docs), docs, 1000)
    print(sim)


def test_doc2vec():
    train_model_doc2vec()
    print("ok", 'doc2vec')
    load_model_doc2vec()

if __name__ == '__main__':
    test_doc2vec()
    test_word2vec()