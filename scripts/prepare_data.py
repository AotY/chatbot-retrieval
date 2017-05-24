import os
import csv
import itertools
import functools
import tensorflow as tf
import numpy as np
import array

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
  "input_dir", os.path.abspath("./data"),
  "Input directory containing original CSV data files (default = './data')")

tf.flags.DEFINE_string(
  "output_dir", os.path.abspath("./data"),
  "Output directory for TFrEcord files (default = './data')")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.csv")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "valid.csv")
TEST_PATH = os.path.join(FLAGS.input_dir, "test.csv")

#返回一个
def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

def create_csv_iter(filename):
  """
  Returns an iterator over a CSV file. Skips the header.
  """
  with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header
    next(reader)
    for row in reader:
      yield row


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

  vocab_processor.fit(input_iter)
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
  context, utterance, label = row
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  label = int(float(label))

  # New Example
  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
  example.features.feature["label"].int64_list.value.extend([label])
  return example


def create_example_test(row, vocab):
  """
  Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  context, utterance = row[:2]
  distractors = row[2:]
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)

  # New Example
  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])

  # Distractor sequences
  for i, distractor in enumerate(distractors):
    dis_key = "distractor_{}".format(i)
    dis_len_key = "distractor_{}_len".format(i)
    # Distractor Length Feature
    dis_len = len(next(vocab._tokenizer([distractor])))
    example.features.feature[dis_len_key].int64_list.value.extend([dis_len])
    # Distractor Text Feature
    dis_transformed = transform_sentence(distractor, vocab)
    example.features.feature[dis_key].int64_list.value.extend(dis_transformed)
  return example

'''
# Create train.tfrecords
  create_tfrecords_file(
      input_filename=TRAIN_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
      example_fn=functools.partial(create_example_train, vocab=vocab)
    )
'''
def create_tfrecords_file(input_filename, output_filename, example_fn):
  """
  Creates a TFRecords file for the given input data and
  example transofmration function
  """
  writer = tf.python_io.TFRecordWriter(output_filename) #A class to write records to a TFRecords file.
  print("Creating TFRecords file at {}...".format(output_filename))

  for i, row in enumerate(create_csv_iter(input_filename)):
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
      word =  vocab_processor.vocabulary_._reverse_mapping[id]
      vocabfile.write(word + "\n")
  print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
  print("Creating vocabulary...")

  #文件迭代器
  input_iter = create_csv_iter(TRAIN_PATH)

  input_iter = (x[0] + " " + x[1] for x in input_iter)

  # Create vocabulary.txt file
  vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)

  print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))
  write_vocabulary(
    vocab, os.path.join(FLAGS.output_dir, "vocabulary.txt"))

  # Save vocab processor
  vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

  # Create validation.tfrecords
  create_tfrecords_file(
      input_filename=VALIDATION_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  # Create test.tfrecords
  create_tfrecords_file(
      input_filename=TEST_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  # Create train.tfrecords
  '''
  functools.partial 通过包装手法，允许我们 "重新定义" 函数签名
  用一些默认参数包装一个可调用对象,返回结果是可调用对象，并且可以像原始对象一样对待
  '''
  create_tfrecords_file(
      input_filename=TRAIN_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
      example_fn=functools.partial(create_example_train, vocab=vocab))



