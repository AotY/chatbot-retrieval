import tensorflow as tf
import sys


def get_id_feature(features, key, len_key, max_len):
    ids = features[key]
    ids_len = tf.squeeze(features[len_key], [1]) #Removes dimensions of size 1 from the shape of a tensor.
    ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64)) #ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64)) #

    return ids, ids_len


def create_train_op(loss, hparams):
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=hparams.learning_rate,
        clip_gradients=10.0,
        optimizer=hparams.optimizer)
    return train_op


# 创建模型方法，返回model_fn()方法
def create_model_fn(hparams, model_impl):
    def model_fn(features, targets, mode):

        question, question_len = get_id_feature(
            features, "question", "question_len", hparams.max_question_len)

        answer, answer_len = get_id_feature(
            features, "answer", "answer_len", hparams.max_answer_len)

        if targets is not None:
            batch_size = targets.get_shape().as_list()[0]

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            probs, mean_loss = model_impl(
                hparams,
                mode,
                question,
                question_len,
                answer,
                answer_len,
                targets)
            train_op = create_train_op(mean_loss, hparams)
            return probs, mean_loss, train_op

        if mode == tf.contrib.learn.ModeKeys.INFER:
            probs, mean_loss = model_impl(
                hparams,
                mode,
                question,
                question_len,
                answer,
                answer_len,
                None)
            return probs, 0.0, None

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            all_questions = [question]
            all_question_lens = [question_len]
            all_answers = [answer]
            all_answer_lens = [answer_len]
            # all_targets = [tf.ones([batch_size, 1], dtype=tf.int64)]
            all_targets = [targets]
            probs, mean_loss = model_impl(
                hparams,
                mode,
                tf.concat(all_questions, 0),
                tf.concat(all_question_lens, 0),
                tf.concat(all_answers, 0),
                tf.concat(all_answer_lens, 0),
                tf.concat(all_targets, 0)
            )
            split_probs = tf.split(probs, 1, 0)
            shaped_probs = tf.concat(split_probs, 1)

            tf.summary.histogram("eval_correct_probs_hist", split_probs[0])
            tf.summary.scalar("eval_correct_probs_average", tf.reduce_mean(split_probs[0]))

            return shaped_probs, mean_loss, None

        # if mode == tf.contrib.learn.ModeKeys.EVAL:
        #     probs, mean_loss = model_impl(
        #         hparams,
        #         mode,
        #         question,
        #         question_len,
        #         answer,
        #         answer_len,
        #         targets)
        #
        #     tf.summary.histogram("eval_correct_probs_hist", probs)
        #     tf.summary.scalar("eval_correct_probs_average", tf.reduce_mean(probs))
        #
        #     return probs, mean_loss



        # with tf.Session().as_default():
        # print('probs: ', tf.shape(probs))
        # print('mean_loss:  ', tf.shape(mean_loss))

        # split_probs = tf.split(probs, 10, 0) # Caused by op 'split', defined at:
        # split_probs = tf.split(probs, 1, 0) # Caused by op 'split', defined at:
        # split_probs = tf.split(probs, 2, 0) # Caused by op 'split', defined at:
        # shaped_probs = tf.concat(split_probs, 1)

        # Add summaries
        # tf.summary.merge_all()
        # tf.summary.histogram("eval_correct_probs_hist", probs)
        # tf.summary.scalar("eval_correct_probs_average", tf.reduce_mean(probs))


        # tf.summary.histogram("eval_incorrect_probs_hist", split_probs[1])
        # tf.summary.scalar("eval_incorrect_probs_average", tf.reduce_mean(split_probs[1]))

        # tf.histogram_summary("eval_correct_probs_hist", split_probs[0])
        # tf.scalar_summary("eval_correct_probs_average", tf.reduce_mean(split_probs[0]))
        # tf.histogram_summary("eval_incorrect_probs_hist", split_probs[1])
        # tf.scalar_summary("eval_incorrect_probs_average", tf.reduce_mean(split_probs[1]))



    return model_fn
