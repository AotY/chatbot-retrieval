import os
import time
import itertools
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model

# tf.flags.DEFINE_string("input_dir", "./data",
#                        "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("input_dir", "./data/BoP2017_DBAQ_dev_train_data/",
                       "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")

tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 2, "Evaluate after this many train steps")
# tf.flags.DEFINE_string("test", None, "description")
FLAGS = tf.flags.FLAGS

# 时间戳, 用于生成保存模型的路径
TIMESTAMP = int(time.time())

if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))

# 训练数据集和训练验证数据集
TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.tfrecords"))

# 设置日志等级
tf.logging.set_verbosity(FLAGS.loglevel)

'''
Because we have different formats of training and evaluation data I’ve written a
create_model_fn wrapper that takes care of bringing the data into the right format for us.
It takes a model_impl argument, which is a function that actually makes predictions.
In our case it’s the Dual Encoder LSTM we described above, but we could easily swap
it out for some other neural network. Let’s see what that looks like:
'''


def main(unused_argv):
    # 获取udc_hparams文件设置参数
    hparams = udc_hparams.create_hparams()

    # 设置模型方法，传递参数和dual_encoder_model
    model_fn = udc_model.create_model_fn(
        hparams,
        model_impl=dual_encoder_model)

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIR,
        config=tf.contrib.learn.RunConfig())

    '''
    1. Create a feature definition that describes the fields in our Example file
    2. Read records from the input_files with tf.TFRecordReader
    3. Parse the records according to the feature definition
    4. Extract the training labels
    5. Batch multiple examples and training labels
    6. Return the batched examples and training labels
    '''

    input_fn_train = udc_inputs.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        input_files=[TRAIN_FILE],
        batch_size=hparams.batch_size,
        num_epochs=FLAGS.num_epochs)

    input_fn_eval = udc_inputs.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.EVAL,
        input_files=[VALIDATION_FILE],
        batch_size=hparams.eval_batch_size,
        num_epochs=1)

    eval_metrics = udc_metrics.create_evaluation_metrics()

    """Runs evaluation of a given estimator, at most every N steps.

      Note that the evaluation is done based on the saved checkpoint, which will
      usually be older than the current step.

      Can do early stopping on validation metrics if `early_stopping_rounds` is
      provided.
      """
    eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_eval,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics)

    estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])


if __name__ == "__main__":
    tf.app.run()
    # tf.train.SessionRunHook()
