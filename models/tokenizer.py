# coding: UTF-8
import tensorflow as tf
from pybloom import BloomFilter


tf.app.flags.DEFINE_string("data_dir", './../../data/BoP2017_DBAQ_dev_train_data/', "stopword.")

FLAGS = tf.app.flags.FLAGS

# 结巴分词进行初始化
import os

import jieba


def init_jieba():
    # 加载用户词典
    jieba.load_userdict(os.path.join(FLAGS.data_dir, "userdict.txt"))
    pass


init_jieba()

bloomFilter = BloomFilter(capacity=1000, error_rate=0.001)


# 加载停顿词
def load_stop_word():
    with open(os.path.join(FLAGS.data_dir, "stopword.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            bloomFilter.add(line.rstrip())


load_stop_word()


# 分词
def tokenizer_fn(sentence):
    # return (x.split(" ") for x in iterator)
    # # 精确模式 HMM 参数用来控制是否使用 HMM 模型  于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法

    # seg_list = jieba.cut(x, cut_all=False, HMM=True)  # 精确模式
    seg_list = jieba.cut(sentence, cut_all=True)  # 全模式
    # seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    # print('seg_list', seg_list)
    no_stop_list = remove_stop(seg_list)
    return no_stop_list


# 去除停顿词
def remove_stop(seg_list):
    return [word for word in seg_list if word not in bloomFilter]
