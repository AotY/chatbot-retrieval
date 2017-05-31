import codecs
import os
import functools
from pybloom import BloomFilter
import tensorflow as tf
import jieba

'''
The dataset originally comes in CSV format.
We could work directly with CSVs, but it’s better to convert our data
into Tensorflow’s proprietary Example format.

(Quick side note: There’s also tf.SequenceExample but it doesn’t seem
to be supported by tf.learn yet).

The main benefit of this format is that it allows us to load tensors directly
from the input files and let Tensorflow handle all the shuffling, batching and
queuing of inputs.

As part of the preprocessing we also create a vocabulary.
This means we map each word to an integer number, e.g. “cat” may become 2631.
The TFRecord files we will generate store these integer numbers instead of the word strings.
We will also save the vocabulary so that we can map back from integers to words later on.
'''
#
tf.flags.DEFINE_integer(
    "min_word_frequency", 5, "Minimum frequency of words in the vocabulary")

#
tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string(
    "input_dir", os.path.abspath("./../data/BoP2017_DBAQ_dev_train_data/"),
    "Input directory containing original CSV data files (default = './data/BoP2017_DBAQ_dev_train_data/')")

tf.flags.DEFINE_string(
    "output_dir", os.path.abspath("./../data/BoP2017_DBAQ_dev_train_data/"),
    "Output directory for TFrEcord files (default = './..data/BoP2017_DBAQ_dev_train_data/')")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.txt")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "dev.txt")


# TEST_PATH = os.path.join(FLAGS.input_dir, "test.csv")



# 结巴分词进行初始化
def init_jieba():
    # 加载用户词典
    jieba.load_userdict(os.path.join(FLAGS.input_dir, "userdict.txt"))
    pass


init_jieba()

bloomFilter = BloomFilter(capacity=1000, error_rate=0.001)


# 加载停顿词
def load_stop_word():
    with codecs.open(os.path.join(FLAGS.input_dir, "stopword.txt"), 'rb', encoding='utf-8') as f:
        for line in f:
            bloomFilter.add(line.rstrip())


load_stop_word()


# 分词
def tokenizer_fn(iterator):
    # return (x.split(" ") for x in iterator)
    # # 精确模式 HMM 参数用来控制是否使用 HMM 模型  于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法
    for x in iterator:
        # seg_list = jieba.cut(x, cut_all=False, HMM=True)
        seg_list = jieba.cut(x, cut_all=True, HMM=True) #精确模式
        # seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
        # print('seg_list', seg_list)
        no_stop_list = remove_stop(seg_list)
        yield no_stop_list


# 去除停顿词
def remove_stop(seg_list):
    return [word for word in seg_list if word not in bloomFilter]


def create_txt_iter(filename):
    with codecs.open(filename, encoding='utf-8') as file:
        for line in file:
            line = line.rstrip()
            words = line.split('\t')
            yield words


'''
构建词典，所有训练集中出现的单词
'''


def create_vocab(input_iter, min_frequency):
    """
    Creates and returns a VocabularyProcessor object with the vocabulary
    for the input iterator.
    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_sentence_len,
        min_frequency=min_frequency,
        tokenizer_fn=tokenizer_fn)

    vocab_processor.fit(input_iter)  # input_iter: An iterable which yield either str or unicode.
    return vocab_processor


def transform_sentence(sequence, vocab_processor):
    """
    Maps a single sentence into the integer vocabulary. Returns a python array.
    """
    return next(vocab_processor.transform([sequence])).tolist()


def create_text_sequence_feature(fl, sentence, sentence_len, vocab):
    """
    Writes a sentence to FeatureList protocol buffer
    """
    sentence_transformed = transform_sentence(sentence, vocab)
    for word_id in sentence_transformed:
        fl.feature.add().int64_list.value.extend([word_id])
    return fl


def create_example_train(row, vocab):
    """
    Creates a training example for the Ubuntu Dialog Corpus dataset.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    label, question, anwser = row

    question_transformed = transform_sentence(question, vocab)
    anwser_transformed = transform_sentence(anwser, vocab)

    question_len = len(next(vocab._tokenizer([question])))
    anwser_len = len(next(vocab._tokenizer([anwser])))

    label = label.rstrip()
    label = label[-1]
    # print(label)
    label = int(float(label))

    # New Example
    example = tf.train.Example()
    example.features.feature["question"].int64_list.value.extend(question_transformed)
    example.features.feature["anwser"].int64_list.value.extend(anwser_transformed)
    example.features.feature["question_len"].int64_list.value.extend([question_len])
    example.featlquestionabelures.feature["anwser_len"].int64_list.value.extend([anwser_len])
    example.features.feature["lquestionabel"].int64_list.value.extend([label])
    return example  # 返回一个样例


def create_tfrecords_file(input_filename, output_filename, example_fn):
    """
    Creates a TFRecords file for the given input data and
    example transofmration function
    """
    writer = tf.python_io.TFRecordWriter(output_filename)  # A class to write records to a TFRecords file.
    print("Creating TFRecords file at {}...".format(output_filename))

    for i, row in enumerate(create_txt_iter(input_filename)):
        x = example_fn(row)
        writer.write(x.SerializeToString())

    writer.close()
    print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):
    """
    Writes the vocabulary to a file, one word per line.
    """
    vocab_size = len(vocab_processor.vocabulary_)
    with open(outfile, "w") as vocabfile:
        for id in range(vocab_size):
            word = vocab_processor.vocabulary_._reverse_mapping[id]
            vocabfile.write(word + "\n")
    print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
    print("Creating vocabulary...")

    # 文件迭代器
    input_iter = create_txt_iter(TRAIN_PATH)

    # 文件列表 label, question, anwser
    input_iter = (x[1].rstrip() + " " + x[2].rstrip() for x in input_iter)

    # Create vocabulary.txt file
    vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)

    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))
    write_vocabulary(
        vocab, os.path.join(FLAGS.output_dir, "vocabulary.txt"))

    # Save vocab processor
    vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

    # Create train.tfrecords
    '''
    functools.partial 通过包装手法，允许我们 "重新定义" 函数签名
    用一些默认参数包装一个可调用对象,返回结果是可调用对象，并且可以像原始对象一样对待
    '''
    create_tfrecords_file(
        input_filename=TRAIN_PATH,
        output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
        example_fn=functools.partial(create_example_train, vocab=vocab)
    )

    # Create validation.tfrecords
    create_tfrecords_file(
        input_filename=VALIDATION_PATH,
        output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"),
        example_fn=functools.partial(create_example_train, vocab=vocab)
    )
