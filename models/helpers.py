import array
import numpy as np
import tensorflow as tf
from collections import defaultdict
import codecs


def load_vocab(filename):
    vocab = None
    with open(filename, encoding='utf-8') as f:
        vocab = f.read().splitlines()
    dct = defaultdict(int)
    for idx, word in enumerate(vocab):
        dct[word] = idx
    return [vocab, dct]


def load_word2vec_vectors(filename, vocab):
    encoding = 'utf-8'
    dct = {}
    vectors = array.array('d')
    current_idx = 0
    with open(filename, "rb") as f:
        header = f.readline()
        print('header :  ', header)
        vocab_size, vector_size = map(int, header.split())  # 头信息， 记录vocabulary_size and vector_size eg: '314143 300'
        binary_len = np.dtype('float32').itemsize * vector_size
        for line_no in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                if ch != b'\n':
                    word.append(ch)
            word = str(b''.join(word), encoding=encoding, errors='strict')
            vector = np.fromstring(f.read(binary_len), dtype='float32')
            word = word.rstrip()

            if not vocab or word in vocab:
                dct[word] = current_idx
                vectors.extend(float(x) for x in vector)
                current_idx += 1

        word_dim = len(vector)
        num_vectors = len(dct)

        tf.logging.info("Found {} out of {} vectors in word2vec".format(num_vectors, len(vocab)))
        print("Found {} out of {} vectors in word2vec".format(num_vectors, len(vocab)))
        return [np.array(vectors).reshape(num_vectors, word_dim), dct]



def load_glove_vectors(filename, vocab):
    """
    Load glove vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """
    dct = {}
    vectors = array.array('d')
    current_idx = 0

    with open(filename, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not vocab or word in vocab:
                dct[word] = current_idx
                vectors.extend(float(x) for x in entries)
                current_idx += 1

        word_dim = len(entries)
        num_vectors = len(dct)

        tf.logging.info("Found {} out of {} vectors in Glove".format(num_vectors, len(vocab)))
        print("Found {} out of {} vectors in Glove".format(num_vectors, len(vocab)))
        return [np.array(vectors).reshape(num_vectors, word_dim), dct]
        # glove_vectors, glove_dict


def load_fastText_vectors(filename, vocab):
    """
    Load fastText vectors from wiki.zh.vec file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """
    encoding = 'utf-8'
    dct = {}
    vectors = array.array('d')
    current_idx = 0
    with open(filename, "rb") as f:
        header = f.readline()

        vocab_size, vector_size = map(int, header.split())  # 头信息， 记录vocabulary_size and vector_size eg: '314143 300'
        print('vocab_size: {} ,   vector_size: {}'.format(vocab_size, vector_size))

        for line_no in range(vocab_size):
            line = f.readline()
            if line == b'':
                raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
            parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
            if len(parts) != vector_size + 1:
                raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
            # word, vector = parts[0], list(map('float32', parts[1:]))
            word, vector = parts[0], list(map(np.float, parts[1:]))
            # idx = vocabulary.get(word)
            # if idx != 0:
            #     embedding_vectors[idx] = vector
            #
            if not vocab or word in vocab:
                dct[word] = current_idx
                vectors.extend(float(x) for x in vector)
                current_idx += 1

        word_dim = len(vector)
        num_vectors = len(dct)

        tf.logging.info("Found {} out of {} vectors in fastText".format(num_vectors, len(vocab)))
        print("Found {} out of {} vectors in fastText".format(num_vectors, len(vocab)))
        return [np.array(vectors).reshape(num_vectors, word_dim), dct]



'''
vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim
一个文本就是一个matrix，
'''

def build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, embedding_dim):
    initial_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")
    for word, glove_word_idx in glove_dict.items():
        word_idx = vocab_dict.get(word)
        initial_embeddings[word_idx, :] = glove_vectors[glove_word_idx]

    return initial_embeddings

if __name__ == '__main__':
    pass
