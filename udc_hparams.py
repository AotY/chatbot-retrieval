import tensorflow as tf
from collections import namedtuple

'''
hparams is a custom object we create in hparams.py that holds hyperparameters, 
nobs we can tweak, of our model. 
This hparams object is given to the model when we instantiate it.
'''
# Model Parameters
tf.flags.DEFINE_integer(
    "vocab_size",
    # 91620,
    61353,
    "The size of the vocabulary. Only change this if you changed the preprocessing")

# Model Parameters
# tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")

# using pretrained vector
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of the embeddings")

tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_question_len", 160, "Truncate questions to this length")
tf.flags.DEFINE_integer("max_anwser_len", 80, "Truncate anwser to this length")

# Pre-trained embeddings

tf.flags.DEFINE_string("glove_path", 'data/pre-trained/glove.840B.300d.txt', \
                       "Path to pre-trained Glove vectors")

tf.flags.DEFINE_string("word2vec_path", 'data/pre-trained/GoogleNews-vectors-negative300.bin', \
                       "Path to pre-trained word2vec vectors")

tf.flags.DEFINE_string("fastText_path", None, "Path to pre-trained fastText vectors")

# 这个应该是训练文本里面出现的词汇集合
# tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")

tf.flags.DEFINE_string("vocab_path", 'data/BoP2017_DBAQ_dev_train_data/vocabulary.txt', \
                       "Path to vocabulary.txt file")

tf.flags.DEFINE_string("vector_type", 'word2vec', 'word2vec or glove or fastText')
# tf.flags.DEFINE_string("vector_type", 'glove', 'word2vec or glove or fastText')

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")

# tf.flags.DEFINE_integer("batch_size", 128, "Batch size during training")
# tf.flags.DEFINE_integer("eval_batch_size", 16, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
    "HParams",
    [
        "batch_size",
        "embedding_dim",
        "eval_batch_size",
        "learning_rate",
        "max_question_len",
        "max_anwser_len",
        "optimizer",
        "rnn_dim",
        "vocab_size",
        "glove_path",
        "word2vec_path",
        "fastText_path",
        "vector_type",
        "vocab_path"
    ])


def create_hparams():
    return HParams(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        vocab_size=FLAGS.vocab_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        embedding_dim=FLAGS.embedding_dim,
        max_question_len=FLAGS.max_question_len,
        max_anwser_len=FLAGS.max_anwser_len,
        glove_path=FLAGS.glove_path,
        word2vec_path=FLAGS.word2vec_path,
        fastText_path=FLAGS.fastText_path,
        vector_type=FLAGS.vector_type,
        vocab_path=FLAGS.vocab_path,
        rnn_dim=FLAGS.rnn_dim)
