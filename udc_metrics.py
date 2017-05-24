import tensorflow as tf
import functools
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec


'''
We already mentioned that we want to use the recall@k metric to evaluate our model.
Luckily, Tensorflow already comes with many standard evaluation metrics that we can use, including recall@k.
To use these metrics we need to create a dictionary that maps from a metric name to a function that takes the
predictions and label as arguments:


'''
def create_evaluation_metrics():
    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        eval_metrics["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(
            tf.contrib.metrics.streaming_sparse_recall_at_k,
            k=k))
    return eval_metrics
