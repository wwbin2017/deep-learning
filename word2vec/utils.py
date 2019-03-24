#!/usr/bin/python
# --*-- coding: utf-8 --*--

import re
import os
import collections


def findPart(text, regex=u"[\w\u2E80-\u9FFF]+"):
    res = re.findall(regex, unicode(text,'utf8'))
    word = ""
    if res and len(res) == 1:
        word = res[0]
    return word


def read_file(fp):
    '''
    读取文件，空格划分，保存到list中
    :param fp: 打开的文件流
    :return: 返回单词列表 
    '''
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    # step 1 读取停用词
    stop_words = []
    with open('stop_words.txt') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1].decode("utf-8"))
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

    # step2 读取文本，预处理
    words = []
    line = fp.readline()
    while line:
        # print line
        if len(line) > 0:  # 如果句子非空
            raw_words = line.split(" ")
            for word in raw_words:
                word = findPart(word)
                if word and word.decode("utf-8") not in stop_words:
                    flag = 1
                    for wo in word:
                        wo = wo.decode("utf-8")
                        if wo in stop_words:
                            flag = 0
                            break
                    if flag == 1:
                        words.append(word)
        line = fp.readline()

    print('文本中总共有{n1}个单词'.format(n1=len(words)))
    return words


def read_dir(data_path='./'):
    # data_path = "wikidata/wikiParser/wiki_output/wiki"
    file_list = []
    files=os.listdir(data_path)
    for file_name in files:
        file_list.append(data_path+'/'+file_name)
    return file_list


def docs2list(data_path):
    file_list = read_dir(data_path)
    docs = []
    for file_name in file_list:
        with open(file_name) as fp:
            words = read_file(fp)
            docs.append(words)
    return docs

def build_dataset(words, vocabulary_size=50000):
    '''
    根据单词列表，构建字典
    :param words: 单词列表
    :param vocabulary_size: 字典大小
    :return: 
        data: words对应的字典序号
        count： 词和出现的次数
        dictionary： 字典,（key，value):(词，序号)
        reverse_dictionary： 字典,（key，value):(序号,词)
    '''
    count = []
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    count.append(['UNK', -1])
    dictionary = dict()  # {word: index}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()  # collect index
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = vocabulary_size-1  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[len(count)-1][1] = unk_count  # list of tuples (word, count)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def build_doc_dataset(docs, vocabulary_size=50000):
    '''
    构建doc数据集
    :param docs: 文档列表，每个元素为一个list，一个list表示一篇文章
    :param vocabulary_size: 字典大小
    :return:doc_ids: 文档id
            word_ids: 词id
            count： 词和出现的次数
            dictionary： 字典,（key，value):(词，序号)
            reverse_dictionary： 字典,（key，value):(序号,词)
    '''
    words = []
    doc_ids = []  # collect document(sentence) indices
    for i, doc in enumerate(docs):
        doc_ids.extend([i] * len(doc))
        words.extend(doc)

    word_ids, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size=vocabulary_size)

    return doc_ids, word_ids, count, dictionary, reverse_dictionary

