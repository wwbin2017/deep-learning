#!/usr/bin/python
# --*-- coding: utf-8 --*--

from __future__ import absolute_import
from __future__ import print_function

import os
import math
import json
import collections
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import compress

import numpy as np
import tensorflow as tf

import utils


data_index = 0


def generate_batch_pvdm(doc_ids, word_ids, batch_size, window_size):
    '''
    产生 PV-DM (Distributed Memory Model of Paragraph Vectors) 训练数据.
    :param doc_ids: 文档id
    :param word_ids: 词id
    :param batch_size: 批训练大小
    :param window_size: 窗口大小
    :return: batch：训练数据
            labels：标签
    '''
    global data_index
    assert batch_size % window_size == 0
    batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = window_size + 1
    buffer = collections.deque(maxlen=span)
    buffer_doc = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)

    mask = [1] * span
    mask[-1] = 0
    i = 0
    while i < batch_size:
        if len(set(buffer_doc)) == 1:
            doc_id = buffer_doc[-1]
            batch[i, :] = list(compress(buffer, mask)) + [doc_id]
            labels[i, 0] = buffer[-1]
            i += 1
        # 滑动窗口
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)

    return batch, labels


class Doc2Vec(BaseEstimator, TransformerMixin):
    '''
    训练doc2vec模型，用于计算文档或者段落向量
    '''

    def __init__(self, batch_size=128, window_size=8,
                 concat=True,
                 embedding_size_w=128,
                 embedding_size_d=128,
                 vocabulary_size=50000,
                 document_size=100,
                 n_neg_samples=64,
                 learning_rate=1e-4, n_steps=100001):
        # 模型变量初始化
        self.batch_size = batch_size
        self.window_size = window_size
        self.concat = concat
        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.vocabulary_size = vocabulary_size
        self.document_size = document_size
        self.n_neg_samples = n_neg_samples
        self.learning_rate = learning_rate
        self.n_steps = n_steps

        self._choose_batch_generator()
        # 构建图，初始化变量
        self._init_graph()

        # 创建 session
        self.sess = tf.Session(graph=self.graph)

    def _choose_batch_generator(self):
        self.generate_batch = generate_batch_pvdm

    def _init_graph(self):
        '''
        构建图
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size + 1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            self.word_embeddings = tf.Variable(   # 词向量矩阵
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))
            self.doc_embeddings = tf.Variable(    # 文档向量矩阵
                tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0))

            if self.concat:  # 拼接词向量，再和拼接doc拼接
                combined_embed_vector_length = self.embedding_size_w * self.window_size + self.embedding_size_d
            else:  # 词向量相加求均值，再和拼接doc拼接
                combined_embed_vector_length = self.embedding_size_w + self.embedding_size_d

            # 每个词对应的softmax权重
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],
                                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            embed = []  # 输入词对应的词向量 shape=(batch_size, embedding_size)
            if self.concat:
                for j in range(self.window_size):
                    embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                    embed.append(embed_w)
            else:
                # 求词向量的和
                embed_w = tf.zeros([self.batch_size, self.embedding_size_w])
                for j in range(self.window_size):
                    embed_w += tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                embed.append(embed_w)

            embed_d = tf.nn.embedding_lookup(self.doc_embeddings, self.train_dataset[:, self.window_size])
            embed.append(embed_d)
            # 拼接词向量和文档向量
            # self.embed = tf.concat(1, embed)
            self.embed = tf.concat(embed, 1)
            # 损失函数
            loss = tf.nn.nce_loss(weights=self.weights,
                                  biases=self.biases,
                                  inputs=self.embed,
                                  labels=self.train_labels,
                                  num_sampled=self.n_neg_samples,
                                  num_classes=self.vocabulary_size)
            self.loss = tf.reduce_mean(loss)
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # 归一化
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / norm_w
            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / norm_d
            # 变量初始化
            self.init_op = tf.global_variables_initializer()
            # 模型存储
            self.saver = tf.train.Saver()

    def _build_dictionaries(self, docs):
        '''
        数据集构建
        :param docs: 文档集
        :return: doc_ids： 文档映射后的id
                word_ids： 词映射后的id
        '''
        doc_ids, word_ids, count, dictionary, reverse_dictionary = utils.build_doc_dataset(docs,
                                                                                     self.vocabulary_size)
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.count = count
        return doc_ids, word_ids

    def train_model(self, docs):
        '''
        模型训练
        :param docs: 文档集 
        :return: 返回模型
        '''
        doc_ids, word_ids = self._build_dictionaries(docs)
        session = self.sess
        session.run(self.init_op)
        average_loss = 0
        print("变量初始化完成")
        for step in range(self.n_steps):
            batch_data, batch_labels = self.generate_batch(doc_ids, word_ids,
                                                           self.batch_size, self.window_size)
            feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
            op, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # 两千次的迭代，平均损失
                print('第 %d 步的平均损失: %f' % (step, average_loss))
                average_loss = 0
        # 得到词向量的归一化的结果
        self.word_embeddings = session.run(self.normalized_word_embeddings)
        self.doc_embeddings = session.run(self.normalized_doc_embeddings)

        return self

    def save(self, path):
        '''
        保存模型和参数
        '''
        if os.path.isfile(path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(path):
            os.mkdir(path)
        tf_path = os.path.join(path, 'model.ckpt')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        save_path = self.saver.save(self.sess,
                                    os.path.join(path, 'model.ckpt'))

        params = self.get_params()
        json.dump(params,
                  open(os.path.join(path, 'model_params.json'), 'wb'))
        json.dump(self.dictionary,
                  open(os.path.join(path, 'model_dict.json'), 'wb'),
                  ensure_ascii=False)
        json.dump(self.reverse_dictionary,
                  open(os.path.join(path, 'model_rdict.json'), 'wb'),
                  ensure_ascii=False)

        print("模型保存到: %s" % save_path)
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path+'/model.ckpt')

    @classmethod
    def restore(cls, path):
        '''
        加载模型
        '''
        params = json.load(open(os.path.join(path, 'model_params.json'), 'rb'))

        estimator = Doc2Vec(**params)
        estimator._restore(path)
        # word 和 doc 向量
        estimator.word_embeddings = estimator.sess.run(estimator.normalized_word_embeddings)
        estimator.doc_embeddings = estimator.sess.run(estimator.normalized_doc_embeddings)

        estimator.dictionary = json.load(open(os.path.join(path, 'model_dict.json'), 'rb'))
        reverse_dictionary = json.load(open(os.path.join(path, 'model_rdict.json'), 'rb'))

        estimator.reverse_dictionary = {int(key): val for key, val in reverse_dictionary.items()}
        return estimator

    def predict_doc_predict(self, document_size, docs, n_steps):
        '''
        构建图
        '''
        graph = tf.Graph()
        with graph.as_default():
            train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size + 1])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            word_embeddings = self.word_embeddings   # 词向量矩阵
            doc_embeddings = tf.Variable(  # 文档向量矩阵
                tf.random_uniform([document_size, self.embedding_size_d], -1.0, 1.0))

            # 每个词对应的softmax权重
            weights = self.sess.run(self.weights)
            biases = self.sess.run(self.biases)

            embed = []  # 输入词对应的词向量 shape=(batch_size, embedding_size)
            if self.concat:
                for j in range(self.window_size):
                    embed_w = tf.nn.embedding_lookup(word_embeddings, train_dataset[:, j])
                    embed.append(embed_w)
            else:
                # 求词向量的和
                embed_w = tf.zeros([self.batch_size, self.embedding_size_w])
                for j in range(self.window_size):
                    embed_w += tf.nn.embedding_lookup(word_embeddings, train_dataset[:, j])
                embed.append(embed_w)

            embed_d = tf.nn.embedding_lookup(doc_embeddings, train_dataset[:, self.window_size])
            embed.append(embed_d)
            # 拼接词向量和文档向量
            embed_concat = tf.concat(embed, 1)
            # 损失函数
            loss = tf.nn.nce_loss(weights=weights,
                                  biases=biases,
                                  inputs=embed_concat,
                                  labels=train_labels,
                                  num_sampled=self.n_neg_samples,
                                  num_classes=self.vocabulary_size)
            loss = tf.reduce_mean(loss)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # 归一化
            norm_d = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
            normalized_doc_embeddings = doc_embeddings / norm_d
            # 变量初始化
            init_op = tf.global_variables_initializer()

            # 模型训练
            doc_ids, word_ids = self.doc2id(docs)
            session = tf.Session(graph=graph)
            session.run(init_op)

            average_loss = 0
            print("初始化完成！")
            for step in range(n_steps):
                batch_data, batch_labels = self.generate_batch(doc_ids, word_ids,
                                                               self.batch_size, self.window_size)
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                op, l = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if step % 20 == 0:
                    if step > 0:
                        average_loss = average_loss / 20
                    # 2000词的平均损失
                    print('在 %d 步的平均损失: %f' % (step, average_loss))
                    average_loss = 0

            doc_embeddings = session.run(normalized_doc_embeddings)

        return doc_embeddings

    def doc2id(self, docs):
        words = []
        doc_ids = []  # collect document(sentence) indices
        word_ids = []
        for i, doc in enumerate(docs):
            doc_ids.extend([i] * len(doc))
            words.extend(doc)
        for word in words:
            if word in self.dictionary:
                word_ids.append(self.dictionary[word])
            else:
                word_ids.append(self.dictionary['UNK'])
        return doc_ids, word_ids

