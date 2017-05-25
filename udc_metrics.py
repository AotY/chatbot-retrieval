import tensorflow as tf
import functools
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec


'''
We already mentioned that we want to use the recall@k metric to evaluate our model.
Luckily, Tensorflow already comes with many standard evaluation metrics that we can use, including recall@k.
To use these metrics we need to create a dictionary that maps from a metric name to a function that takes the
predictions and label as arguments:

资格赛MRR评分计算
1. 为保证结果公正，程序会对每个答案集合里相等的实数值进行随机排序。
在这种情况下，同一结果提交两次，MRR评分会不同。

2. 此程序是计算模型在开发集数据上的表现。
'''
def create_evaluation_metrics():
    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        eval_metrics["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(
            tf.contrib.metrics.streaming_sparse_recall_at_k,
            k=k))
    return eval_metrics
