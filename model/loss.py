import tensorflow as tf
import numpy as np
from model.common import shape_list, l2_normalize, pairwise_euc_distances


# TODO: add additive margin softmax, additive angular margin softmax


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
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer
    with tf.variable_scope("softmax", reuse=reuse_variables):
        logits = tf.layers.dense(features,
                                   num_outputs,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer),
                                   name="output")
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
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
        # L2 normalization
        with tf.name_scope("length_norm"):
            features = l2_normalize(features)

        # There are 2 variables in the End2End loss
        w = tf.get_variable("w", initializer=float(params.init_end2end_w), dtype=tf.float32)
        b = tf.get_variable("b", initializer=float(params.init_end2end_b), dtype=tf.float32)

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
                index[i * num_segments_per_speaker + j, 0] = i * num_segments_per_speaker + j
                index[i * num_segments_per_speaker + j, 1] = i

        if params.ge2e_loss_type == "softmax":
            # loss = -tf.reduce_sum(similarity_true - tf.reduce_logsumexp(similarity, axis=1))
            # Use tf sparse_softmax_cross_entropy to compute the softmax value more efficiently.
            loss = tf.losses.sparse_softmax_cross_entropy(index[:, 1], similarity)
        elif params.ge2e_loss_type == "contrastive":
            similarity_true = tf.gather_nd(similarity, index)
            similarity_neg = tf.reduce_max(tf.multiply(1-mask, tf.sigmoid(similarity)), axis=1)
            loss = tf.truediv(tf.reduce_sum(1 - tf.sigmoid(similarity_true) + similarity_neg),
                              tf.to_float(num_features))
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
    with tf.variable_scope("triplet_loss", reuse=reuse_variables):
        # L2 normalization
        with tf.name_scope("length_norm"):
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


def asoftmax(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
    """Compute angular softmax.
    SphereFace: Deep Hypersphere Embedding for Face Recognition
    https://arxiv.org/abs/1704.08063

    Args:
        features: A tensor with shape [batch, dim].
        labels: A tensor with shape [batch].
        num_outputs: The number of classes.
        params: params.weight_l2_regularizer: the L2 regularization.
                params.asoftmax_m: The m value.
                params.asoftmax_lambda_min,
                params.asoftmax_lambda_base,
                params.asoftmax_lambda_gamma,
                params.asoftmax_lambda_power,
                params.asoftmax_norm, params.asoftmax_s: If asoftmax_norm is True, asoftmax_s must be specified.
                                                         This means we normalize the length of the features, and do the
                                                         scaling on the cosine similarity.
                params.global_step:  All used to tune lambda.
        is_training: Not used in this case
        reuse_variables: Reuse variables.
    :return: The A-softmax loss.
    """
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer
    # We keep the name of the variable_scope as the same with softmax loss to enable fine-tuning.
    with tf.variable_scope("softmax", reuse=reuse_variables):
        # There are 1 variables in angular softmax: the weight matrix w.
        # The name of w is selected as the same with softmax to enable fine-tuning.
        w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
        norm = tf.norm(features, axis=1, keep_dims=True)
        features_norm = tf.nn.l2_normalize(features, axis=1)
        w_norm = tf.nn.l2_normalize(w, axis=0)

        # cos(theta)
        cos_theta = tf.matmul(features_norm, w_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)

        # Feature normalization and scaling if necessary
        if "asoftmax_norm" in params.dict and params.asoftmax_norm:
            assert "asoftmax_s" in params.dict, "If feature normalization is applied, scaling factor is necessary."
            # logits = s * cos(theta)
            logits = params.asoftmax_s * cos_theta
        else:
            # logits = ||x||*cos(theta)
            logits = norm * cos_theta

        if params.asoftmax_m == 1:
            # logits = ||x|| * cos(theta) or s * cos(theta)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            return loss

        ordinal = tf.to_int32(tf.range(shape_list(features)[0]))
        ordinal_labels = tf.stack([ordinal, labels], axis=1)
        # The angle between x and the target w_i.

        # cos(theta_i), theta_i is the angle between x_i and its target w_i.
        cos_theta_i = tf.gather_nd(cos_theta, ordinal_labels)

        # Phi(theta, m) = (-1)^k * cos(2*theta) - 2 * k
        # Phi(theta_i, m) is a monotonical function about theta given m
        if params.asoftmax_m == 2:
            phi = 2 * tf.multiply(tf.sign(cos_theta_i), tf.square(cos_theta_i)) - 1
        elif params.asoftmax_m == 4:
            cos_th2 = tf.square(cos_theta_i)
            cos_th4 = tf.pow(cos_theta_i, 4)
            sign0 = tf.sign(cos_theta_i)
            sign3 = tf.multiply(tf.sign(2 * cos_th2 - 1), sign0)
            sign4 = 2 * sign0 + sign3 - 3
            phi = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4
        else:
            raise NotImplementedError("[ERROR] m=%d is not unsupported." % params.asoftmax_m)

        asoftmax_lambda = tf.maximum(float(params.asoftmax_lambda_min),
                                     params.asoftmax_lambda_base * (1.0 + params.asoftmax_lambda_gamma * tf.to_float(params.global_step)) ** (-params.asoftmax_lambda_power))
        tf.summary.scalar("asoftmax_lambda", asoftmax_lambda)
        fa = 1.0 / (1.0 + asoftmax_lambda)
        fs = 1.0 - fa
        cos_theta_asoftmax = tf.add(cos_theta,
                                    tf.scatter_nd(ordinal_labels,
                                                  tf.subtract(phi, cos_theta_i),
                                                  tf.shape(cos_theta, out_type=tf.int32)))
        updated_cos_theta = fs * cos_theta + fa * cos_theta_asoftmax

        if "asoftmax_norm" in params.dict and params.asoftmax_norm:
            updated_logits = params.asoftmax_s * updated_cos_theta
        else:
            # ||x|| * Phi(theta, m)
            updated_logits = norm * updated_cos_theta

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=updated_logits)
        return loss


def additive_margin_softmax(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
    """Additive margin softmax.
    link: https://arxiv.org/abs/1801.05599
    ref: https://github.com/Joker316701882/Additive-Margin-Softmax

    Args:
        features: A tensor with shape [batch, dim].
        labels: A tensor with shape [batch].
        num_outputs: The number of classes.
        params: params.weight_l2_regularizer: the L2 regularization.
                params.amsoftmax_m: the margin. (0.25-0.5)
                params.amsoftmax_norm, params.amsoftmax_s: If amsoftmax_norm is True, amsoftmax_s must be specified.
                                                         This means we normalize the length of the features, and do the
                                                         scaling on the cosine similarity.
        is_training: Not used in this case.
        reuse_variables: Reuse variables.
    """
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer
    with tf.variable_scope("softmax", reuse=reuse_variables):
        w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
        norm = tf.norm(features, axis=1, keep_dims=True)
        features = tf.nn.l2_normalize(features, axis=1)
        w_norm = tf.nn.l2_normalize(w, axis=0)
        cos_theta = tf.matmul(features, w_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)  # for numerical steady
        phi = cos_theta - params.amsoftmax_m
        labels_onehot = tf.one_hot(labels, num_outputs, 1, 0, dtype=tf.int32)

        if "amsoftmax_norm" in params.dict and params.amsoftmax_norm:
            assert "amsoftmax_s" in params.dict, "If feature normalization is applied, scaling factor is necessary."
            # s * (cos(theta) - m)
            logits_amsoftmax = params.amsoftmax_s * tf.where(tf.equal(labels_onehot, 1), phi, cos_theta)
        else:
            logits_amsoftmax = norm * tf.where(tf.equal(labels_onehot, 1), phi, cos_theta)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_amsoftmax)
        return loss


# def ring_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None):


def additive_angular_margin_softmax(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
    """Additive angular margin softmax (ArcFace)
    link: https://arxiv.org/abs/1801.07698

    Args:
        features: A tensor with shape [batch, dim].
        labels: A tensor with shape [batch].
        num_outputs: The number of classes.
        params: params.weight_l2_regularizer: the L2 regularization.
                params.arcsoftmax_m: the angular margin (0.4-0.55)
                params.arcsoftmax_norm, params.arcsoftmax_s: If arcsoftmax_norm is True, arcsoftmax_s must be specified.
                                                         This means we normalize the length of the features, and do the
                                                         scaling on the cosine similarity.
        is_training: Not used in this case.
        reuse_variables: Reuse variables.
    """
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer
    with tf.variable_scope("softmax", reuse=reuse_variables):
        w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
        norm = tf.norm(features, axis=1, keep_dims=True)
        features = tf.nn.l2_normalize(features, axis=1)
        w_norm = tf.nn.l2_normalize(w, axis=0)

        cos_theta = tf.matmul(features, w_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)

        ordinal = tf.to_int32(tf.range(shape_list(features)[0]))
        ordinal_labels = tf.stack([ordinal, labels], axis=1)
        # The angle between x and the target w_i.
        cos_theta_i = tf.gather_nd(cos_theta, ordinal_labels)

        # Since 0 < theta < pi, sin(theta) > 0. sin(theta) = sqrt(1 - cos(theta)^2)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sin_theta_i_sq = 1 - tf.square(cos_theta_i)
        mask = tf.to_float(tf.less(sin_theta_i_sq, 1e-16))
        sin_theta_i = tf.sqrt(sin_theta_i_sq + mask * 1e-16) * (1 - mask)
        cos_theta_plus_m_i = cos_theta_i * tf.cos(params.arcsoftmax_m) - sin_theta_i * tf.sin(params.arcsoftmax_m)

        # Since theta \in [0, pi], theta > pi + m means cos(theta) < cos(pi+m)
        # If theta < pi - m, Phi(theta) = cos(theta + m).
        # If theta > pi + m, Phi(theta) = -cos(theta + m) - 2
        phi = tf.where(tf.greater(cos_theta_i, tf.cos(np.pi + params.arcsoftmax_m)),
                       cos_theta_plus_m_i,
                       -cos_theta_plus_m_i - 2)

        cos_arcsoftmax = tf.add(cos_theta,
                                tf.scatter_nd(ordinal_labels,
                                              phi - cos_theta_i, tf.shape(cos_theta, out_type=tf.int32)))

        if "arcsoftmax_norm" in params.dict and params.arcsoftmax_norm:
            assert "arcsoftmax_s" in params.dict, "If feature normalization is applied, scaling factor is necessary."
            # s * (cos(theta) - m)
            logits_arcsoftmax = params.arcsoftmax_s * cos_arcsoftmax
        else:
            logits_arcsoftmax = norm * cos_arcsoftmax
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_arcsoftmax)
        return loss


def test_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
    """This is used to test a new loss. """
    endpoints = {}
    with tf.variable_scope("ge2e", reuse=reuse_variables):
        embeddings_input = features
        # There are 2 variables in the End2End loss
        w = tf.get_variable("w", initializer=float(params.init_end2end_w), dtype=tf.float32)
        b = tf.get_variable("b", initializer=float(params.init_end2end_b), dtype=tf.float32)

        tf.summary.scalar("w", w)
        tf.summary.scalar("b", b)

        # The inputs contain N speakers and M segments per speaker.
        num_features = shape_list(features)[0]
        dim = shape_list(features)[1]
        num_segments_per_speaker = params.num_segments_per_speaker
        num_speakers = params.num_speakers_per_batch
        assert num_segments_per_speaker != 1

        # L2 normalization
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

        ######################################################
        # Another methods to compute the loss

        norm_out = tf.nn.l2_normalize(embeddings_input, dim=-1)

        def _cal_centroid_matrix(utt_idx):
            def cal_centroid(centroid_idx):
                # utt_idx counts from 0 to 639
                # spk_id counts from 0 to 63
                spk_id = (utt_idx // num_segments_per_speaker)
                utt_idx_in_group = utt_idx % num_segments_per_speaker

                mask = tf.equal(tf.scatter_nd([[utt_idx_in_group]], tf.constant([1]),
                                       shape=[num_segments_per_speaker]), 0)
                all_utts_for_spk = norm_out[centroid_idx * num_segments_per_speaker: (centroid_idx + 1) * num_segments_per_speaker, :]
                centroid = tf.cond(tf.equal(centroid_idx, spk_id),
                                   lambda: tf.reduce_mean(tf.boolean_mask(all_utts_for_spk, mask), 0),
                                   lambda: tf.reduce_mean(all_utts_for_spk, 0))
                return centroid

            centroid_mat = tf.convert_to_tensor(
                tf.map_fn(cal_centroid, tf.range(num_speakers), dtype=tf.float32))
            return centroid_mat

        def tf_scaled_cosine_similarity(ia, ib):
            # returns similarity vector of utt for every centroid
            normalize_a = tf.reshape(tf.nn.l2_normalize(ia, dim=-1), [1, -1])
            normalize_b = tf.transpose(tf.nn.l2_normalize(ib, dim=-1))

            # cosine similarity vector [1,64]
            cos_similarity = tf.reshape(tf.matmul(normalize_a, normalize_b), [-1])  # [1,64] to [64]

            # scaled cosine similarity
            scaled_cos_similarity = tf.add(tf.multiply(w, cos_similarity), b)

            return scaled_cos_similarity

        def _create_sim_per_utt(utt_idx):
            # utt_dvector is a tensor of shape [output_size]
            utt_dvector = norm_out[utt_idx, :]
            # centroids is a tensor of shape [num_spk_per_batch, output_size]
            # sim_per_utt is a tensor of shape [num_spk_per_batch]
            centroids = _cal_centroid_matrix(utt_idx)
            sim_per_utt = tf_scaled_cosine_similarity(utt_dvector, centroids)
            return sim_per_utt

        sim_mat = tf.convert_to_tensor(
            tf.map_fn(_create_sim_per_utt, tf.range(num_features), dtype=tf.float32))

        # Check sim_mat

        target_batch = np.zeros((num_segments_per_speaker*num_speakers,), dtype=np.int32)
        for i in range(num_speakers):
            target_batch[i*(num_segments_per_speaker):(i+1)*num_segments_per_speaker] = i
        loss_2 = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sim_mat, labels=target_batch))

        return loss, loss_2
