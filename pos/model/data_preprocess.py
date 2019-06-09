#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import os
import re

# 数据保存的地址
save_path = './dictionary'


def save2json(data, save_path, name):
    """
    把字典数据类型保存到json里面
    :param data:  数据
    :param save_path: 路径 
    :param name: 文件名
    :return: 无
    """
    if os.path.isfile(save_path):
        raise RuntimeError('the save path should be a dir')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    json.dump(data,
              open(os.path.join(save_path, name), 'wb'),
              ensure_ascii=False)


def build_dataset(words, dict_name, rdict_name):
    """
    
    :param words: 
    :param dict_name: 
    :param rdict_name: 
    :return: dictionary： 字典,（key，value):(词，序号)
        reverse_dictionary： 字典,（key，value):(序号,词)
    """
    count = set(words)
    dictionary = dict()  # {word: index}
    for word in count:
        dictionary[word] = len(dictionary) + 1
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # 保存字典到json格式
    save2json(dictionary, save_path, dict_name)
    save2json(reverse_dictionary, save_path, rdict_name)
    return dictionary


def char2id(sentence, dictionary):
    """
    把词转化为相应的id
    :param sentence: 二维的list， 第一维表示句子， 第二维表示每个句子的每个词 
    :param dictionary: 字典
    :return: 句子的每个词的id
    """
    sen_id = []
    for sen in sentence:
        senId = []
        for word in sen:
            if word in dictionary.keys():
                senId.append(dictionary[word])
            else:
                senId.append(0)
        sen_id.append(senId)
    return sen_id


def read_file(file_name):
    """
    :param file_name: 
    :return: 
    """
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    # step1 读取文本，预处理
    data_words = []
    data_labels = []
    words = set()
    labels = set()
    fp = open(file_name, 'r')
    line = fp.readline()
    i = 0
    while line:
        if i > 1000:
            break
        i = i + 1
        if len(line) > 0:
            word = []
            label = []
            line = unicode(line, "utf-8")
            raw_words = line.replace("\n", "").split(" ")
            for w in raw_words:
                word_label = w.split('/')
                if len(word_label) == 2:
                    unicode_word = word_label[0].replace("[", "").replace("]", "")
                    word.append(unicode_word)
                    words.add(unicode_word)
                    labels.add(word_label[1])
                    label.append(word_label[1])
                    print(unicode_word)
                    if unicode_word[len(unicode_word)-1] in [u',', u'.', u'!', u';', u'?',
                                                             u'，', u'。', u'！', u'？', u'；']:
                        data_words.append(word)
                        data_labels.append(label)
                        word = []
                        label = []
        if len(word) > 0:
            data_words.append(word)
            data_labels.append(label)
        line = fp.readline()
    fp.close()
    return data_words, data_labels, words, labels


def data2id(file_name):
    """
    字符转id
    :param file_name: 
    :return: 
    """
    data_words, data_labels, words, labels = read_file(file_name)
    word_num = len(words)
    label_num = len(labels)
    words_dict = build_dataset(words, 'model_dict.json', 'model_rdict.json')
    labels_dict = build_dataset(labels, 'label_dict.json', 'label_rdict.json')
    word_id = char2id(data_words, words_dict)
    label_id = char2id(data_labels, labels_dict)
    return word_id, label_id, word_num, label_num


if __name__ == '__main__':
    file_name = "../data/people_2014_01.txt"
    data2id(file_name)


