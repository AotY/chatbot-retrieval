import tensorflow as tf

TEXT_FEATURE_SIZE = 160


def get_feature_columns(mode):
    feature_columns = []

    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="question", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="question_len", dimension=1, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="anwser", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="anwser_len", dimension=1, dtype=tf.int64))

    feature_columns.append(tf.contrib.layers.real_valued_column(column_name='label', dimension=1, dtype=tf.int64))

    # if mode == tf.contrib.learn.ModeKeys.TRAIN:
    #     # During training we have a label feature
    #     feature_columns.append(tf.contrib.layers.real_valued_column(
    #         column_name="label", dimension=1, dtype=tf.int64))
    #
    # if mode == tf.contrib.learn.ModeKeys.EVAL:
    #     # During evaluation we have distractors
    #     for i in range(9):
    #       feature_columns.append(tf.contrib.layers.real_valued_column(
    #         column_name="distractor_{}".format(i), dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
    #       feature_columns.append(tf.contrib.layers.real_valued_column(
    #         column_name="distractor_{}_len".format(i), dimension=1, dtype=tf.int64))

    return set(feature_columns)


'''
1. Create a feature definition that describes the fields in our Example file
2. Read records from the input_files with tf.TFRecordReader
3. Parse the records according to the feature definition
4. Extract the training labels
5. Batch multiple examples and training labels
6. Return the batched examples and training labels

'''

'''

All queue runners are added to the queue runners collection, and may be started
via start_queue_runners.
All ops are added to the default graph.

Adds operations to read, queue, batch and parse Example protos.
Given file pattern (or list of files), will setup a queue for file names, read Example proto
using provided reader, use bath queue to create batches of examples of size batch_size and parse
example given features specification.

return A dict of Tensor or SparseTensor objects for each in features.

'''
def create_input_fn(mode, input_files, batch_size, num_epochs):
    def input_fn():
        features = tf.contrib.layers.create_feature_spec_for_parsing(
            get_feature_columns(mode))

        feature_map = tf.contrib.learn.io.read_batch_features(
            file_pattern=input_files,
            batch_size=batch_size,
            features=features,
            reader=tf.TFRecordReader,
            randomize_input=True,
            num_epochs=num_epochs,
            queue_capacity=200000 + batch_size * 10,
            name="read_batch_features_{}".format(mode))

        # This is an ugly hack because of a current bug in tf.learn
        # During evaluation TF tries to restore the epoch variable which isn't defined during training
        # So we define the variable manually here
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            tf.get_variable(
                "read_batch_features_eval/file_name_queue/limit_epochs/epochs",
                initializer=tf.constant(0, dtype=tf.int64))

        target = feature_map.pop('label')
        # if mode == tf.contrib.learn.ModeKeys.TRAIN:
        #   target = feature_map.pop("label")
        # else:
        #   # In evaluation we have 10 classes (utterances).
        #   # The first one (index 0) is always the correct one
        #   target = tf.zeros([batch_size, 1], dtype=tf.int64)
        return feature_map, target

    return input_fn
