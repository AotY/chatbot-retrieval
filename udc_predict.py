import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model
from models.helpers import load_vocab

tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/BoP2017_DBAQ_dev_train_data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)


# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load your own data here
INPUT_question = "香港会议展览中心会展2期的屋顶的是由什么建成的，形状是什么？"
POTENTIAL_RESPONSES = ["香港会议展览中心（简称会展；英语：Hong Kong Convention and Exhibition Centre，缩写：HKCEC）是香港的主要大型会议及展览场地，位于香港岛湾仔北岸，是香港地标之一；由香港政府及香港贸易发展局共同拥有，由新创建集团的全资附属机构香港会议展览中心（管理）有限公司管理。", "会展2期的屋顶以4万平方呎的铝合金造成，形状像是一只飞鸟。"]

def get_features(question, anwser):
  question_matrix = np.array(list(vp.transform([question])))
  anwser_matrix = np.array(list(vp.transform([anwser])))
  question_len = len(question.split(" "))
  anwser_len = len(anwser.split(" "))
  features = {
    "question": tf.convert_to_tensor(question_matrix, dtype=tf.int64),
    "question_len": tf.constant(question_len, shape=[1,1], dtype=tf.int64),
    "anwser": tf.convert_to_tensor(anwser_matrix, dtype=tf.int64),
    "anwser_len": tf.constant(anwser_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

# 提交时间：2017年6月6日12:00am前
if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(
    tf.constant(0, shape=[1,1])
  )

  print("question: {}".format(INPUT_question))
  for r in POTENTIAL_RESPONSES:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_question, r))
    print("{}: {:g}".format(r, prob[0,0]))