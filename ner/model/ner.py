#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
ner model
"""

from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import json
import re

import data_pre

index_data = 0

file_name_chinese = "../data/corpusData/NER/chinese_data.txt"
file_name_english = "../data/corpusData/NER/english_data.txt"
print("Data processing begins ......")
word_id, label_id, words_dict, labels_dict = data_pre.data2id(file_name_chinese, file_name_english)
print("dictionary size: ", len(words_dict), "   label size: ", len(labels_dict))
print("Data processing ends ......")
print("Model training begins ......")


def get_batch(batch_size, time_step, word_id, label_id, label_size):
    """
    产生批量训练数据
    :param batch_size: 
    :param time_step: 
    :param word_id: 
    :param label_id: 
    :param label_size: 
    :return: train_word_id 维度 [batch_size][time_step]
            train_label_id [batch_size][time_step][label_size]
    """

    global index_data
    train_word_id = np.zeros(shape=[batch_size, time_step])
    train_label_id = np.zeros(shape=[batch_size, time_step, label_size])
    index = index_data
    for i in range(batch_size):
        index = index % len(word_id)
        while len(word_id[index]) > time_step:
            index = (index + 1) % len(word_id)
        for j in range(len(word_id[index])):
            train_word_id[i, j] = word_id[index][j]
            train_label_id[i, j, label_id[index][j]] = 1
        for j in range(len(word_id[index]), time_step):
            train_label_id[i, j, 0] = 1
        # 下一个句子
        index = index + 1
    index_data = index
    return train_word_id, train_label_id


def data_tranform(time_step, word_id):
    """
    产生批量训练数据
    :param time_step: 
    :param word_id: 
    :return: train_word_id 维度 [batch_size][time_step] 
    """
    train_word_id = np.zeros(shape=[len(word_id), time_step])
    for i in range(len(word_id)):
        for j in range(len(word_id[i])):
            if j < time_step:
                train_word_id[i, j] = word_id[i][j]
    return train_word_id


class ConfigArgs(object):
    """NER参数配置"""
    label_size = 22  # 类别数
    init_scale = 0.01  # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
    learning_rate = 1e-3  # 学习速率,在文本循环次数超过max_epoch以后会逐渐降低
    max_grad_norm = 10  # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
    num_layers = 1  # lstm层数
    num_steps = 100  # 单个数据中，序列的长度。
    hidden_size = 512  # 隐藏层中单元数目
    keep_prob = 0.5  # 用于dropout.每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
    batch_size = 8  # 每批数据的规模，每批有10个。
    max_loop = 20000  # 运行次数
    charNum = 10000  # 词典大小
    char_dim = 512   # 每个字符嵌入的维度
#    hide_dim = 200    # 隐层维度


class NER(object):
    """
    NER
    """

    def __init__(self, config, is_training, embedding_char=None):
        """
        初始化函数
        :param config: 模型的配置参数 
        :param embedding_char: 训练好的词向量
        """
        self.config = config
        self.is_training = is_training
        self.embedding_char = embedding_char
        #############
        self.config.label_size = len(labels_dict) + 1
        self.config.charNum = len(words_dict) + 1
        #############
        # 初始化
        self.sess = None
        self.graph = None
        self.init = None
        self.losses = None
        self._train_op = None
        self.saver = None
        # 构建图，初始化
        self.build_graph()
        self.init_op()

    def init_op(self):
        """
        分词初始化
        :return: 无
        """
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def build_graph(self):
        """
        模型构建
        """
        self.graph = tf.Graph()

        with self.graph.as_default():
            # 维度 [batch][char_id]
            self.char_id = tf.placeholder(tf.int32, shape=[None, self.config.num_steps])
            # 维度 [batch][char_id][label_size]
            self.labels = tf.placeholder(tf.float32, shape=[None, self.config.num_steps, self.config.label_size])

            def lstm_cell():
                return tf.contrib.rnn.GRUCell(self.config.hidden_size)

            attn_cell = lstm_cell
            if self.is_training and self.config.keep_prob < 1:  # dropout 防止过拟合
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(
                        lstm_cell(), output_keep_prob=self.config.keep_prob)

            cell_1_forward = tf.contrib.rnn.MultiRNNCell(  # 多层lstm
                [attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
            cell_1_backward = tf.contrib.rnn.MultiRNNCell(  # 多层lstm
                [attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)

            cell_2_forward = tf.contrib.rnn.MultiRNNCell(  # 多层lstm
                [attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
            cell_2_backward = tf.contrib.rnn.MultiRNNCell(  # 多层lstm
                [attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)

            # 嵌入向量
            with tf.device("/cpu:0"):
                # 获取词嵌入
                if self.embedding_char is not None:
                    embedding_char = tf.Variable(self.embedding_char,
                                                 trainable=True, name="embedding_char", dtype=tf.float32)
                else:
                    embedding_char = tf.Variable(
                        tf.truncated_normal([self.config.charNum, self.config.char_dim], stddev=0.1),
                        trainable=True, name="embedding_char", dtype=tf.float32)
                inputs_char = tf.nn.embedding_lookup(embedding_char, self.char_id)

            # BiLSTM
            outputs_1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                cell_1_forward, cell_1_backward, tf.unstack(inputs_char, self.config.num_steps, 1),
                dtype=tf.float32, scope='bidirectional_rnn_0')
            # dropout
            if self.is_training and self.config.keep_prob < 1:
                outputs_1 = tf.nn.dropout(outputs_1, self.config.keep_prob)  # 维度 [step][None][feature*2]

            # BiLSTM
            outputs_2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                cell_2_forward, cell_2_backward, tf.unstack(outputs_1, self.config.num_steps, 0),
                dtype=tf.float32, scope='bidirectional_rnn_1')

            # dropout
            if self.is_training and self.config.keep_prob < 1:
                outputs_2 = tf.nn.dropout(outputs_2, self.config.keep_prob)  # 维度 [step][None][feature*2]
            else:
                outputs_2 = tf.stack(outputs_2)

            weight = tf.Variable(tf.random_uniform([self.config.hidden_size*2, self.config.label_size], -0.1, 0.1))
            bias = tf.Variable(tf.zeros([self.config.label_size]))

            # 预测结果保存
            y_pre_sum = []
            for t_s in range(self.config.num_steps):
                y_pre = tf.nn.softmax(
                    tf.matmul(outputs_2[t_s, :, :], weight) + bias
                )
                y_pre_sum.append(y_pre)  # 维度 [step][batch_size][labels]
                cross_entropy = -tf.reduce_sum(self.labels[:, t_s, :] * tf.log(y_pre + 1e-10))
                tf.add_to_collection('losses', cross_entropy)

            # 结果 [step][batch_size]
            self.correct_prediction = tf.argmax(tf.stack(y_pre_sum), 2)

            # 模型评估
            for each_ in range(len(y_pre_sum)):
                correct_prediction = tf.equal(tf.argmax(y_pre_sum[each_], 1),
                                              tf.argmax(self.labels[:, each_, :], 1))
                tf.add_to_collection('evaluate', tf.reduce_mean(tf.cast(correct_prediction, "float")))
            self.evaluate = tf.add_n(tf.get_collection('evaluate'), name="evaluate_loss")/self.config.num_steps

            # 优化损失函数
            self.losses = tf.add_n(tf.get_collection('losses'), name="total_loss")

            self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.losses)
            '''
            # 自适应学习率调准
            self.AdaDelta_train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(self.losses)
            self.Adam_train_op = tf.train.AdamOptimizer(1e-3).minimize(self.losses)
            '''
            #  梯度限制， 防止梯度爆炸
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.losses, tvars),
                                              self.config.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

            # init 和 saver
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def train_model(self):
        """
        模型的训练
        :return: 无 
        """
        # 训练模型
        for i in range(self.config.max_loop):
            train_word_id, train_label_id = get_batch(self.config.batch_size, self.config.num_steps,
                                                      word_id, label_id, self.config.label_size)
            feed_dict = {
                self.char_id: train_word_id,
                self.labels: train_label_id
            }
            loss, _, evaluate = self.sess.run([self.losses, self._train_op, self.evaluate], feed_dict=feed_dict)
            if i % 10 == 0:
                print("迭代次数：", i," 损失：", loss, ' 准确率：', evaluate)
            if i % 100 == 0:
                # 保存模型
                self.save_model(save_path="./model_data")
                print("model save successful")

    def predict(self, train_word_id):
        """
        预测
        :param train_word_id: 预测字符的id 
        :return: # 维度 [step][batch_size]
        """
        feed_dict = {
            self.char_id: train_word_id,
        }
        # 结果 [step][batch_size]
        correct_prediction = self.sess.run(self.correct_prediction, feed_dict=feed_dict)
        return correct_prediction

    def predict_char(self, word, words_dict):
        """
        标注句子，分词
        :param word: 需分词的句子
        :param words_dict: 字符字典
        :return: 返回每个字符的 label id
                    维度 [step][batch_size]
        """
        char_id = data_pre.char2id(word, words_dict)
        char_id = data_tranform(self.config.num_steps, char_id)
        label_id = self.predict(char_id)
        return label_id

    def id2label(self, word, label_id, labels_rdict):
        """
        将分词的标注转为 对应的标签
        :param word: 需标注的文本
        :param label_id: 标注的文本对应的标签 # 维度 [step][batch_size]
        :param labels_dict: 标签对应的字典
        :return: labels  dim is [batch_size][word]
        """
        labels = []
        for i in range(len(word)):
            label = []
            for j in range(len(word[i])):
                if j < self.config.num_steps and label_id[j][i] in labels_rdict.keys():
                    label.append(labels_rdict[label_id[j][i]])
                else:
                    label.append('o')
            labels.append(label)
        return labels

    def save_model(self, save_path):
        """
        模型的保存
        :param save_path: 保存的路径 
        :return: 返回模型路径
        """
        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # 记录tf模型
        tf_path = os.path.join(save_path, 'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess, tf_path)
        print("模型保存到: %s" % save_path)
        return save_path

    def restore(self, path):
        """
        模型的加载
        :param path: 模型加载路径
        :return: 无
        """
        with self.graph.as_default():
            self.saver.restore(self.sess, path + '/tf_vars')


def load_data(path, name, reverse):
    """
    加载数据
    :param path: 路径 
    :param name: 文件名
    :param reverse: 是否是逆向字典
    :return: 返回对应字典的类型
    """
    if reverse is False:
        dictionary = json.load(open(os.path.join(path, name), 'rb'))
    else:
        reverse_dictionary = json.load(open(os.path.join(path, name), 'rb'))
        dictionary = {int(key): val for key, val in reverse_dictionary.items()}
    return dictionary


def load_model_data():
    """
    加载模型需要的数据
    :return: 模型需要的数据 字典
    """
    path = './dictionary'
    model_dict = load_data(path, 'model_dict.json', False)
    model_rdict = load_data(path, 'model_rdict.json', True)
    label_dict = load_data(path, 'label_dict.json', False)
    label_rdict = load_data(path, 'label_rdict.json', True)
    return model_dict, model_rdict, label_dict, label_rdict


def load_model(model_path="./model_data"):
    """
    加载模型
    :param model_path: 
    :return: 返回模型 
    """
    ws = NER(ConfigArgs, False, None)
    ws.restore(model_path)
    return ws


def save_model(model_path="./model_data"):
    ws = NER(ConfigArgs, True, None)
    ws.train_model()
    # 保存模型
    ws.save_model(save_path=model_path)


def predict(ws, sentence):
    """
    predict ne
    :param ws: 
    :param sentence: 
    :return: 
    """
    char_list = word2list(sentence)
    model_dict, model_rdict, label_dict, label_rdict = load_model_data()
    label_id = ws.predict_char(char_list, model_dict)
    labels = ws.id2label(char_list, label_id, label_rdict)
    return list2json(char_list, labels)


def word2list(sentence):
    """
    把句子转化为相应的字符数组
    :param sentence: 句子
    :return: 字符数组
    """
    regular_expression = u'[,.!?，。！？]'
    sentence = unicode(sentence, "utf-8")
    sen = re.split(regular_expression, sentence)
    char_list = []
    for i in range(len(sen)):
        words = [w for w in sen[i]]
        char_list.append(words)
    return char_list


def list2json(sentence, char_list):
    """
    convert predicted result to json
    :param sentence: 
    :param char_list: 
    :return: 
    """
    result = []
    for i in range(len(char_list)):
        word = []
        w = ''
        k = 0
        for j in range(len(char_list[i])):
            if char_list[i][j][-1] in ['s', 'o']:
                if w.strip() != '':
                    temp = dict()
                    temp['id'] = k
                    temp['cont'] = w
                    temp['ne'] = char_list[i][j-1].split('_')[0]
                    k = k + 1
                    word.append(temp)
                temp = dict()
                temp['id'] = k
                temp['cont'] = sentence[i][j]
                temp['ne'] = char_list[i][j].split('_')[0]
                word.append(temp)
                k = k + 1
                w = ''
            if char_list[i][j][-1] in ['b', 'm', 'e']:
                w = w + sentence[i][j]
                if char_list[i][j][-1] == 'e':
                    temp = dict()
                    temp['id'] = k
                    temp['cont'] = w
                    temp['ne'] = char_list[i][j].split('_')[0]
                    word.append(temp)
                    k = k + 1
                    w = ''
            if j == len(char_list[i])-1 and w.strip() != '':
                temp = dict()
                temp['id'] = k
                temp['cont'] = w
                temp['ne'] = char_list[i][j].split('_')[0]
                word.append(temp)
                word.append(w)
        result.append(word)
    return json.dumps(result)


if __name__ == '__main__':
    # save_model()
    print("Model training ends")
    ner = load_model()
    # sentence = ['我是中国人', '结婚的和没结婚的']
    sentence = '我是中国人。结婚的和没结婚的'
    labels = predict(ner, sentence)
    print(labels)

