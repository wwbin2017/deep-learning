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


def read_file_english(file_name):
    """
    :param file_name: 
    :return: 
    """
    # step1 读取文本，预处理
    data_words = []
    data_labels = []
    words = set()
    labels = set()
    # 正则匹配
    pattern = re.compile(r'./\w+')
    fp = open(file_name, 'r')
    line = fp.readline()
    i = 0
    while line:
        if i > 100:
            break
        i = i + 1
        if len(line) > 0:
            word = []
            label = []
            raw_word = line.replace("\n", "")
            raw_words = pattern.findall(raw_word)
            temp_word = ""
            temp_label = None
            for w in raw_words:
                word_label = w.split('/')
                if len(word_label) == 2:
                    if word_label[0] in [u'"', u'\\', u' ', u';', u'.', u'!', u'?', u'；', u'。', u'！', u'？']:
                        if len(temp_word) > 0:
                            word.append(temp_word)
                            words.add(temp_word)
                            if temp_label:
                                t_label = word_label[1].split("_")
                            else:
                                t_label = temp_label.split("_")
                            labels.add(t_label[0])
                            label.append(t_label[0])
                            temp_word = ""
                    else:
                        temp_word = temp_word + word_label[0]
                    # if it is end symbol
                    if word_label[0] in [u';', u'.', u'!', u'?', u'；', u'。', u'！', u'？']:
                        data_words.append(word)
                        data_labels.append(label)
                        word = []
                        label = []
                temp_label = word_label[1]
        if len(word) > 0:
            data_words.append(word)
            data_labels.append(label)
        line = fp.readline()
    fp.close()
    return data_words, data_labels, words, labels


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
                    word.append(word_label[0])
                    words.add(word_label[0])
                    labels.add(word_label[1])
                    label.append(word_label[1])
                    # if it is end symbol
                    if word_label[len(word_label)-1] in [u';', u'.', u'!', u'?', u'；', u'。', u'！', u'？']:
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


def data2id(file_name_chinese, file_name_english):
    """
    字符转id
    :param file_name: 
    :return: 
    """
    data_words_chinese, data_labels_chinese, words_chinese, labels_chinese =\
        read_file(file_name_chinese)
    data_words_english, data_labels_english, words_english, labels_english =\
        read_file_english(file_name_english)
    # combine chinese and english dict data
    words = []
    words.extend(words_chinese)
    words.extend(words_english)
    labels = []
    labels.extend(labels_chinese)
    labels.extend(labels_english)
    words_dict = build_dataset(words, 'model_dict.json', 'model_rdict.json')
    labels_dict = build_dataset(labels, 'label_dict.json', 'label_rdict.json')
    # combine Chinese and english training data
    data_words = []
    data_words.extend(data_words_chinese)
    data_words.extend(data_words_english)
    data_labels = []
    data_labels.extend(data_labels_chinese)
    data_labels.extend(data_labels_english)
    word_id = char2id(data_words, words_dict)
    label_id = char2id(data_labels, labels_dict)
    return word_id, label_id, words_dict, labels_dict


if __name__ == '__main__':
    file_name_chinese = "../data/corpusData/NER/chinese_data.txt"
    file_name_english = "../data/corpusData/NER/english_data.txt"
    data2id(file_name_chinese, file_name_english)


