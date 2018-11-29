import tensorflow as tf
import numpy as np
from misc.utils import shape_list, l2_normalize


def softmax(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
    """Vanilla softmax loss.

    Args:
        features: A tensor with shape [batch, dim].
        labels: A tensor with shape [batch].
        num_outputs: The number of classes.
        params: params.weight_l2_regularizer used for L2 regularization.
        is_training: Not used in this case
        reuse_variables: Share the created variables or not.
    :return: A scalar tensor which is the loss value.
    """
    with tf.variable_scope("softmax", reuse=reuse_variables):
        features = tf.layers.dense(features,
                                   num_outputs,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="output")
        loss = tf.losses.sparse_softmax_cross_entropy(labels, features)
    return loss


def ge2e(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
    """Generalized End2End loss.

    Note the features can be un-normalized.

    Warning: The function require the features are arrange according to speaker, like:
             [s1, s1, s1, s1, s2, s2, s2, s2, ..., sN, sN, sN, sN].
             It is the user's response to ensure the order. If it is not the case, the loss will be incorrect.

    Args:
        features: A tensor with shape [batch, dim] WITHOUT L2 normalization, where batch = num_speakers * num_segments
        labels: A tensor with shape [batch]. Not used in this case. But the features should be arranged carefully!
        num_outputs: Not used in this case
        params: params.num_speakers_per_batch and params.num_segments_per_speaker are very important.
                Make sure their values are consistent with the features.
                params.init_end2end_w, params.init_end2end_b are the initial values for w and b.
        is_training: Not used in this case
        reuse_variables: Share the created variables or not.
    :return:
    """
    with tf.variable_scope("ge2e", reuse=reuse_variables):
        # There are 2 variables in the End2End loss
        w = tf.get_variable("w", initializer=np.array([float(params.init_end2end_w)], dtype=np.float32), dtype=tf.float32)
        b = tf.get_variable("b", initializer=np.array([float(params.init_end2end_b)], dtype=np.float32), dtype=tf.float32)

        # The inputs contain N speakers and M segments per speaker.
        num_features = shape_list(features)[0]
        dim = shape_list(features)[1]
        # num_segments_per_speaker = tf.reduce_sum(
        #     tf.to_int32(tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))), axis=1)[0]
        # num_speakers = num_features / num_segments_per_speaker
        num_segments_per_speaker = params.num_segments_per_speaker
        num_speakers = params.num_speakers_per_batch
        assert num_segments_per_speaker != 1

        # L2 normalization
        features = l2_normalize(features)
        features_reshape = tf.reshape(features, shape=[num_speakers, num_segments_per_speaker, dim])

        # Centers for each speaker
        center = l2_normalize(tf.reduce_mean(features_reshape, axis=1))
        # Centers that exclude each sample
        center_ex = l2_normalize(tf.reshape(tf.reduce_sum(features_reshape, axis=1, keep_dims=True) - features_reshape,
                                            shape=[num_features, dim])
                                 / (num_segments_per_speaker - 1))

        # Similarity matrix
        similarity = tf.matmul(features, tf.transpose(center))
        # Special similarity that using the centers excluding the sample itself.
        similarity_ex = tf.reduce_sum(features * center_ex, axis=1)

        # Combined the two similarities
        # [0, 1, 2, ..., N-1] repeat num_features times
        label_ref = tf.tile(tf.expand_dims(tf.range(num_speakers), axis=0), [num_features, 1])
        # [0, 0, .., 0, 1, 1, ..., N-1, ..., N-1]
        label_new = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_speakers), axis=1), [1, num_segments_per_speaker]),
                               [num_features, 1])
        # mask[i, j] == 1 when feature i belongs to class j
        mask = tf.to_float(tf.equal(label_new, label_ref))
        similarity = similarity * (1 - mask) + tf.expand_dims(similarity_ex, axis=1) * mask

        similarity = tf.abs(w) * similarity + b

        index = np.zeros((num_speakers * num_segments_per_speaker, 2), dtype=np.int32)
        for i in xrange(num_speakers):
            for j in xrange(num_segments_per_speaker):
                index[i*num_segments_per_speaker+j, 0] = i*num_segments_per_speaker+j
                index[i * num_segments_per_speaker + j, 1] = i
        similarity_true = tf.gather_nd(similarity, index)

        if params.ge2e_loss_type == "softmax":
            loss = -tf.reduce_sum(similarity_true - tf.reduce_logsumexp(similarity, axis=1))
        elif params.ge2e_loss_type == "contrastive":
            similarity_neg = tf.reduce_max(tf.multiply(1-mask, tf.sigmoid(similarity)), axis=1)
            loss = tf.reduce_sum(1 - tf.sigmoid(similarity_true) + similarity_neg)
        else:
            raise ValueError("The GE2E only support softmax and contrastive loss")

        return loss

