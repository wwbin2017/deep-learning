#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import collections
import utils
import os
import json
import random


data_index = 0


def generate_batch(word_ids, batch_size=128, num_skips=2, skip_window=2):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=[batch_size], dtype=np.int32)
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(word_ids[data_index])  # data:ids
        data_index = (data_index + 1) % len(word_ids)

    for i in range(batch_size // num_skips):  # how many num_skips in a batch
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]  # extract the middle word
        for j in range(num_skips):
            while target in targets_to_avoid:  # context中的word，一个只取一次
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)  #
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(word_ids[data_index])  # update the buffer, append the next word to buffer
        data_index = (data_index + 1) % len(word_ids)

    return batch, labels  # batch: ids [batch_size] lebels:ids [batch_size*1]


class Word2Vec(object):
    def __init__(self,
                 batch_size=128,           # 批训练大小
                 skip_window=3,                # 单边窗口长
                 num_skips=2,              #
                 embedding_size=200,       # 词向量的长度
                 vocabulary_size=50000,    # 字典大小
                 num_sampled=64,           # 负采样
                 learning_rate=1e-2,
                 n_steps=100001,            # 训练次数
                 logdir='./model/word2vec/tmp/word2vec'
                 ):
        # 获得模型的基本参数
        self.batch_size = batch_size  # 批训练大小
        self.skip_window = skip_window  # 单边窗口长
        self.num_skips = num_skips
        self.embedding_size = embedding_size  # 词向量的长度
        self.vocabulary_size = vocabulary_size  # 字典大小
        self.num_sampled = num_sampled  # 负采样
        self.learning_rate = learning_rate
        self.n_steps = n_steps  # 训练次数
        self.logdir = logdir
        # 批量数据产生器
        self._choose_batch_generator()

        # 构建图，初始化
        self.build_graph()
        self.init_op()

    def _choose_batch_generator(self):
        self.generate_batch = generate_batch

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    def build_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            )
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                              stddev=1.0/math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # 将输入序列向量化
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)  # batch_size

            # 得到NCE损失
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocabulary_size
                )
            )

            # tensorboard 相关
            tf.summary.scalar('loss', self.loss)
            # 根据 nce loss 来更新梯度和embedding
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # 训练操作

            # 计算与指定若干单词的相似度
            self.test_word_id = tf.placeholder(tf.int32, shape=[None])
            vec_l2_model = tf.sqrt(  # 求各词向量的L2模
                tf.reduce_sum(tf.square(self.embedding_dict), 1, keep_dims=True)
            )

            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.summary.scalar('avg_vec_model', avg_l2_model)

            self.normed_embedding = self.embedding_dict / vec_l2_model  # 对embedding向量正则化
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embedding, transpose_b=True)

            # 变量初始化
            self.init = tf.global_variables_initializer()
            self.merged_summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def _build_dictionaries(self, word_list):
        '''
        数据集构建
        :param docs: 文档集
        :return: doc_ids： 文档映射后的id
                word_ids： 词映射后的id
        '''
        word_ids, count, dictionary, reverse_dictionary = utils.build_dataset(word_list,
                                                                              self.vocabulary_size)
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.count = count
        return word_ids

    def train_model(self, word_list):
        word_ids = self._build_dictionaries(word_list)
        sess = self.sess
        average_loss = 0  # 平均损失
        for step in range(self.n_steps):
            batch_inputs, batch_labels = self.generate_batch(word_ids, self.batch_size, self.num_skips,
                                                             self.skip_window)
            feed_dict = {
                self.train_inputs: batch_inputs,
                self.train_labels: batch_labels
            }
            _, loss_val, summary_str = self.sess.run([self.train_op, self.loss, self.merged_summary_op],
                                                 feed_dict=feed_dict)

            # 训练损失
            average_loss += loss_val
            if step % 1000 == 0:
                self.summary_writer.add_summary(summary_str, step)
                if step > 0:
                    average_loss = average_loss / 1000
                    print("已经处理单词数： {a}, 最近10次损失函数平均值: {b}".format(a=step, b=average_loss))
                average_loss = 0

        # 归一化 词向量
        self.word_embeddings = self.sess.run(self.normed_embedding)

        return self

    def cal_similarity(self, test_word_id_list, top_k=10):
        '''
        计算与指定单词id最相似的词
        :param test_word_id_list:  需要计算相似词的 id
        :param top_k: 返回最相似的词的个数
        :return: test_words, 指定单词id的词
                 near_words, 最相近的词
                 sim_mean, 均值
                 sim_var, 方差
                 sim_matrix_top 最相似的词的相似读
        '''
        # normed_embedding = self.sess.run(self.normed_embedding)
        # print(normed_embedding)
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.test_word_id:test_word_id_list})
        sim_mean = np.mean(sim_matrix)
        sim_var = np.mean(np.square(sim_matrix-sim_mean))
        test_words = []
        near_words = []
        sim_matrix_top = []
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.reverse_dictionary[test_word_id_list[i]])
            nearst_id = (-sim_matrix[i, :]).argsort()[1:top_k+1]
            sim_y = -sim_matrix[i, :]
            sim_y.sort()
            sim_matrix_top.append([-sim for sim in sim_y[1:top_k+1]])
            nearst_word = [self.reverse_dictionary[x] for x in nearst_id]
            near_words.append(nearst_word)
        return test_words, near_words, sim_mean, sim_var, sim_matrix_top

    def cal_similarity_word(self, test_word_list, top_k=10):
        test_word_id_list = []
        for word in test_word_list:
            word_id = self.dictionary[word]
            if word_id:
                test_word_id_list.append(word_id)
        test_words, near_words, sim_mean, sim_var, sim_matrix_top = self.cal_similarity(test_word_id_list, top_k)
        return test_words, near_words, sim_mean, sim_var, sim_matrix_top

    def save_model(self, save_path):
        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # 记录模型各参数
        model = {}
        var_names = ['batch_size',           # 批训练大小
                    'skip_window',                # 单边窗口长
                    'num_skips',              #
                    'embedding_size',       # 词向量的长度
                    'vocabulary_size',    # 字典大小
                    'num_sampled',           # 负采样
                    'learning_rate',
                    'n_steps',            # 训练次数
                    'logdir'
                    ]
        for var in var_names:
            model[var] = eval('self.'+var)

        # 保存模型参数
        param_path = os.path.join(save_path, 'params.json')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path, 'wb') as f:
            json.dump(model, f)
        json.dump(self.dictionary,
                  open(os.path.join(save_path, 'model_dict.json'), 'wb'),
                  ensure_ascii=False)
        json.dump(self.reverse_dictionary,
                  open(os.path.join(save_path, 'model_rdict.json'), 'wb'),
                  ensure_ascii=False)

        # 记录tf模型
        tf_path = os.path.join(save_path,'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess, tf_path)
        print("模型保存到: %s" % save_path)
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path+'/tf_vars')

    @classmethod
    def restore(cls, path):
        '''
        加载模型
        '''
        params = json.load(open(os.path.join(path, 'params.json'), 'rb'))

        estimator = Word2Vec(**params)
        estimator._restore(path)
        # word
        estimator.word_embeddings = estimator.sess.run(estimator.normed_embedding)

        estimator.dictionary = json.load(open(os.path.join(path, 'model_dict.json'), 'rb'))
        reverse_dictionary = json.load(open(os.path.join(path, 'model_rdict.json'), 'rb'))

        estimator.reverse_dictionary = {int(key): val for key, val in reverse_dictionary.items()}
        return estimator


