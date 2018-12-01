import tensorflow as tf
import numpy as np
from model.common import shape_list, l2_normalize, pairwise_euc_distances


# TODO: add triplet loss, angular-softmax, additive margin softmax, additive angular margin softmax


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


def ge2e_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
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
    :return: GE2E loss.
    """
    with tf.variable_scope("ge2e", reuse=reuse_variables):
        # There are 2 variables in the End2End loss
        w = tf.get_variable("w", initializer=np.array([float(params.init_end2end_w)], dtype=np.float32), dtype=tf.float32)
        b = tf.get_variable("b", initializer=np.array([float(params.init_end2end_b)], dtype=np.float32), dtype=tf.float32)

        tf.summary.scalar("w", w)
        tf.summary.scalar("b", b)

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


def triplet_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
    """ Computes the triplet loss with semi-hard negative mining.
        This is already implemented in TF 1.10.
        Use the TF implementation (https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py)

        The loss encourages the positive distances (between a pair of embeddings with
        the same labels) to be smaller than the minimum negative distance among
        which are at least greater than the positive distance plus the margin constant
        (called semi-hard negative) in the mini-batch. If no such negative exists,
        uses the largest negative distance instead.
        See: https://arxiv.org/abs/1503.03832.

        This implementation uses tf.tile which make the matrix HUGE. It may fail when the batch size is too large,

    Args:
        features: A tensor with shape [batch, dim] WITHOUT L2 normalization.
        labels: A tensor with shape [batch].
        num_outputs: Not used in this case.
        params: params.margin: the margin of the samples.
                params.triplet_loss_squared:
        is_training: Not used in this case
        reuse_variables: Not used.
    :return: The semi-hard triplet loss.
    """
    # L2 normalization
    features = l2_normalize(features)

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = tf.shape(labels)
    assert lshape.shape == 1
    batch_size = lshape[0]
    labels = tf.reshape(labels, [lshape[0], 1])
    margin = tf.to_float(params.margin)

    # Build pairwise squared distance matrix.
    distances = pairwise_euc_distances(features, params.triplet_loss_squared)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.logical_not(adjacency)

    # Compute the mask.
    distances_tile = tf.tile(distances, [batch_size, 1])

    # mask: valid negative, f_ap < f_an
    # For i-th block, m(x, y) = 1 when label(x) != label(y) and d_xy  > d_xi
    # The i-th block is for triplet (x, i, y)
    mask = tf.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.greater(
            distances_tile, tf.reshape(
                tf.transpose(distances), [-1, 1])))

    # Before reshape:
    # In i-th block, mask_final(x) means for triplet (x, i, y) exists a negative y that d_xy > d_xi
    # After reshape:
    # mask_final(i, x) means triplet (x, i, y) exists a negative y that d_xy > d_xi
    # After transpose:
    # mask_final(x, i) means triplet (x, i, y) exists a negative y that d_xy > d_xi
    mask_final = tf.reshape(
        tf.greater(
            tf.reduce_sum(
                tf.cast(mask, dtype=tf.float32), 1, keep_dims=True),
            0.0), [batch_size, batch_size])
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    def _masked_maximum(data, mask, dim=1):
        """Computes the axis wise maximum over chosen elements.
        Args:
            data: 2-D float `Tensor` of size [n, m].
            mask: 2-D Boolean `Tensor` of size [n, m].
            dim: The dimension over which to compute the maximum.
        Returns:
            masked_maximums: N-D `Tensor`.
              The maximized dimension is of size 1 after the operation.
        """
        axis_minimums = tf.reduce_min(data, dim, keep_dims=True)
        masked_maximums = tf.reduce_max(tf.multiply(data - axis_minimums, mask), dim,
                                        keep_dims=True) + axis_minimums
        return masked_maximums

    def _masked_minimum(data, mask, dim=1):
        """Computes the axis wise minimum over chosen elements.
        Args:
            data: 2-D float `Tensor` of size [n, m].
            mask: 2-D Boolean `Tensor` of size [n, m].
            dim: The dimension over which to compute the minimum.
        Returns:
            masked_minimums: N-D `Tensor`.
              The minimized dimension is of size 1 after the operation.
        """
        axis_maximums = tf.reduce_max(data, dim, keep_dims=True)
        masked_minimums = tf.reduce_min(tf.multiply(data - axis_maximums, mask), dim,
                                        keep_dims=True) + axis_maximums
        return masked_minimums

    # negatives_outside: smallest D_an where D_an > D_ap.
    # negatives_outside(i, x) is the minimum d_xy when y is negative and d_xy > d_xi
    # After transpose: negatives_outside(x, i) means the minimum d_xy of a valid semi-hard negative for triplet (x, i, y)
    negatives_outside = tf.reshape(
        _masked_minimum(distances_tile, mask), [batch_size, batch_size])
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    # For x, negatives_inside(x) is the maximum d_xy for a valid negative
    negatives_inside = tf.tile(
        _masked_maximum(distances, adjacency_not), [1, batch_size])

    # If (x, i) has no semi-hard negative, use the negative y with a maximum d_xy
    semi_hard_negatives = tf.where(
        mask_final, negatives_outside, negatives_inside)

    # margin + d_xi - min(d_xy), y is a valid semi-hard negative for triplet (x, i, y)
    loss_mat = tf.add(margin, distances - semi_hard_negatives)

    # Next, we should find valid positive (x, i)
    mask_positives = tf.cast(
        adjacency, dtype=tf.float32) - tf.diag(
        tf.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.maximum(tf.reduce_sum(mask_positives), 1e-16)

    # compute loss on the valid triplet (x, i, y) with semi-hard negative (x, y)
    loss = tf.truediv(
        tf.reduce_sum(
            tf.maximum(
                tf.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')

    return loss


