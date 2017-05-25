import tensorflow as tf
import numpy as np

import udc_hparams
from models import helpers
from models.load_pretrained_vector import load_embedding_vectors_word2vec, load_embedding_vectors_glove

FLAGS = tf.flags.FLAGS

'''
The Deep Learning model we will build in this post is called a Dual Encoder LSTM network.
This type of network is just one of many we could apply to this problem and
it’s not necessarily the best one. You can come up with all kinds of Deep Learning architectures
that haven’t been tried yet – it’s an active research area. For example,
the seq2seq model often used in Machine Translation would probably do well on this task.

The reason we are going for the Dual Encoder is because it has been reported to give decent
performance on this data set. This means we know what to expect and can be sure that our
implementation is correct. Applying other models to this problem would be an interesting
project.
'''


def get_embeddings(hparams):
    # 加载词汇表，训练数据中包含的
    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    if hparams.vector_type == 'word2vec':
        word2vec_vectors, word2vec_dict = helpers.load_word2vec_vectors(hparams.word2vec_path, vocab=set(vocab_array))
        initializer = helpers.build_initial_embedding_matrix(vocab_dict, word2vec_dict, word2vec_vectors,
                                                             hparams.embedding_dim)
    elif hparams.vector_type == 'glove':
        # glove_vectors所有出现此的向量； glove_dict记录出现此的位置
        glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
        initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors,
                                                             hparams.embedding_dim)
    elif hparams.vector_type == 'fastText':
        # return load_embedding_vectors_fastText(vocab_array, hparams.glove_path, len(vocab_array))
        pass

        # if hparams.glove_path and hparams.vocab_path:
        #     tf.logging.info("Loading Glove embeddings...")
        #     #加载词汇表，训练数据中包含的
        #     vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
        #     #
        #     glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
        #
        #     initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors,
        #                                                          hparams.embedding_dim)
    else:
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)  # 随机均匀

    return tf.get_variable(
        "word_embeddings",
        shape=[hparams.vocab_size, hparams.embedding_dim],
        initializer=initializer)


def dual_encoder_model(
        hparams,
        mode,
        question,
        question_len,
        anwser,
        anwser_len,
        targets):
    # Initialize embedidngs randomly or with pre-trained vectors if available（替换为提前训练好的 vectors)
    embeddings_W = get_embeddings(hparams)

    # Embed the question and the anwser， embedding_lookup-> Looks up `ids` in a list of embedding tensors.
    question_embedded = tf.nn.embedding_lookup(
        embeddings_W, question, name="embed_question")

    anwser_embedded = tf.nn.embedding_lookup(
        embeddings_W, anwser, name="embed_anwser")

    # Build the RNN
    with tf.variable_scope("rnn") as vs:
        # We use an LSTM Cell
        cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(
            hparams.rnn_dim,
            forget_bias=2.0,
            use_peepholes=True,  #
            state_is_tuple=True)

        # Run the anwser and question through the RNN
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell,
            tf.concat([question_embedded, anwser_embedded], 0),
            sequence_length=tf.concat([question_len, anwser_len], 0),
            dtype=tf.float32)

        encoding_question, encoding_anwser = tf.split(rnn_states.h, 2, 0)

    with tf.variable_scope("prediction") as vs:
        # 训练矩阵, ("rnn_dim", 256, "Dimensionality of the RNN cell")
        M = tf.get_variable("M",
                            shape=[hparams.rnn_dim, hparams.rnn_dim],
                            initializer=tf.truncated_normal_initializer())

        # "Predict" a response: c * M
        generated_response = tf.matmul(encoding_question, M)
        generated_response = tf.expand_dims(generated_response, 2)

        encoding_anwser = tf.expand_dims(encoding_anwser, 2)

        # Dot product between generated response and actual response
        # (c * M) * r
        # logits = tf.batch_matmul(generated_response, encoding_anwser, True)
        logits = tf.matmul(generated_response, encoding_anwser, True)
        logits = tf.squeeze(logits, [2])

        # Apply sigmoid to convert logits to probabilities
        probs = tf.sigmoid(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        # Calculate the binary cross-entropy loss
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    return probs, mean_loss


if __name__ == '__main__':
    hparams = udc_hparams.create_hparams()

    # helpers.load_word2vec_vectors(hparams.word2vec_path, None)
    get_embeddings(hparams)
