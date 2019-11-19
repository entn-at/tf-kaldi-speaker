import tensorflow as tf
from model.common import shape_list, dense_bn_relu, dense, dense_relu, dense_tanh, split_heads, combine_last_two_dimensions, prelu
import sys


VAR2STD_EPSILON = 1e-12

def general_pooling(features, aux_features, endpoints, params, is_training):
    """ A general entry of the pooling layer
    """
    # If you add new pooling layer, modify this code.
    # Statistics pooling
    # [b, l, 1500] --> [b, x]
    if params.pooling_type == "statistics_pooling":
        features = statistics_pooling(features, aux_features, endpoints, params, is_training)
    elif params.pooling_type == "self_attention":
        features = self_attention(features, aux_features, endpoints, params, is_training)
    # elif params.pooling_type == "ghost_vlad":
    #     features = ghost_vlad(features, aux_features, endpoints, params, is_training)
    # elif params.pooling_type == "lde":
    #     features = lde(features, aux_features, endpoints, params, is_training)
    else:
        raise NotImplementedError("Not implement %s pooling" % params.pooling_type)
    endpoints['pooling'] = features
    return features

def statistics_pooling(features, aux_features, endpoints, params, is_training):
    """Statistics pooling
    Note that we need to take care of the zeros in the variance since the sqrt on 0 will lead to NaN.

    Args:
        features: A tensor with shape [batch, length, dim].
        aux_features: Auxiliary input features with shape [batch, length, dim].
        endpoints: Outputs of different parts of the network.
        params:
        is_training:
    :return:
        Statistics pooling result [mean, stddev] with shape [batch, dim].
    """
    with tf.variable_scope("stat_pooling"):
        mean = tf.reduce_mean(features, axis=1, keep_dims=True, name="mean")
        variance = tf.reduce_mean(tf.squared_difference(features, mean), axis=1, keep_dims=True, name="variance")
        mean = tf.squeeze(mean, 1)
        variance = tf.squeeze(variance, 1)

        mask = tf.to_float(tf.less_equal(variance, VAR2STD_EPSILON))
        variance = (1.0 - mask) * variance + mask * VAR2STD_EPSILON
        stddev = tf.sqrt(variance)

        stat_pooling = tf.concat([mean, stddev], 1, name="concat")

    return stat_pooling


def self_attention(features, aux_features, endpoints, params, is_training=None):
    """Self-attention.
    In this implementation, `self` is not accurate because the key and value may come from different nodes.
    Note that the key should be the same length with the value, i.e. no convnet is applied after the key layer, or
    some trimming strategy should be applied before the weighted sum.

    Note: We do not use features in this function. The key and value are specified using params
          and are extracted from endpoints.

    Args:
        features: A tensor with shape [batch, length, dim].
        aux_features: Auxiliary input features with shape [batch, length, dim].
        endpoints: Outputs of different parts of the network. Useful when doing attention.
        params: Parameters for self-attention.
            params.att_key_input: endpoints[params.att_key_input] is used to compute the key.
            params.att_key_num_nodes: #nodes of the network to compute the key.
            params.att_key_network_type: The last layer to compute the key.
                                         In the intermediate layers, affine+bn+relu is usually applied
                                         0: affine
                                         1: affine + relu
                                         2: affine + bn + relu
                                         3: affine + tanh
            params.att_value_input: endpoints[params.att_value_input] is used as the value of the component.
            params.att_value_num_nodes: #nodes of the network to compute the value.
            params.att_value_network_type: The layer layer to compute value (if exists).
            params.att_apply_nonlinear: The nonlinearity is applied after the attention weighted sum (default: false).
            params.att_use_scale: Whether to apply a scaling factor when doing the key*query operation.
            params.att_num_heads: The number of heads in multi-head attention.
            params.att_split_key: Whether to split the key when multi-head attention is used.
            params.att_penalty_term: The coefficient of the penalty term.
        is_training: Used in BN.
    :return:
        Attention result. Also in the statistic format [weighted_mean, weighted_stddev]
    """
    relu = tf.nn.relu
    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        if params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu

    with tf.variable_scope("attention"):
        value_features = endpoints[params.att_value_input]
        key_features = endpoints[params.att_key_input]

        # Key forward
        if len(params.att_key_num_nodes) > 1:
            for index, num_nodes in enumerate(params.att_key_num_nodes[:-1]):
                # The intermediate layers use affine+bn+relu
                key_features = dense_bn_relu(key_features, num_nodes, endpoints, params, is_training, name=("att_key%d" % index))
        # The last layer has different choices
        if params.att_key_network_type == 0:
            key_features = dense(key_features, params.att_key_num_nodes[-1], endpoints, params, is_training,
                                 name=("att_key%d" % (len(params.att_key_num_nodes) - 1)))
        elif params.att_key_network_type == 1:
            key_features = dense_relu(key_features, params.att_key_num_nodes[-1], endpoints, params, is_training,
                                      name=("att_key%d" % (len(params.att_key_num_nodes) - 1)))
        elif params.att_key_network_type == 2:
            key_features = dense_bn_relu(key_features, params.att_key_num_nodes[-1], endpoints, params, is_training,
                                         name=("att_key%d" % (len(params.att_key_num_nodes) - 1)))
        elif params.att_key_network_type == 3:
            key_features = dense_tanh(key_features, params.att_key_num_nodes[-1], endpoints, params, is_training,
                                      name=("att_key%d" % (len(params.att_key_num_nodes) - 1)))

        # Value forward
        if len(params.att_value_num_nodes) > 0:
            if len(params.att_value_num_nodes) > 1:
                for index, num_nodes in enumerate(params.att_value_num_nodes[:-1]):
                    value_features = dense_bn_relu(value_features, num_nodes, endpoints, params, is_training,
                                                   name=("att_value%d" % index))
            if params.att_value_network_type == 0:
                value_features = dense(value_features, params.att_value_num_nodes[-1], endpoints, params, is_training,
                                       name=("att_value%d" % (len(params.att_value_num_nodes) - 1)))
            elif params.att_value_network_type == 1:
                value_features = dense_relu(value_features, params.att_value_num_nodes[-1], endpoints, params, is_training,
                                            name=("att_value%d" % (len(params.att_value_num_nodes) - 1)))
            elif params.att_value_network_type == 2:
                value_features = dense_bn_relu(value_features, params.att_value_num_nodes[-1], endpoints, params, is_training,
                                               name=("att_value%d" % (len(params.att_value_num_nodes) - 1)))
            elif params.att_value_network_type == 3:
                value_features = dense_tanh(value_features, params.att_value_num_nodes[-1], endpoints, params, is_training,
                                            name=("att_value%d" % (len(params.att_value_num_nodes) - 1)))

        # The last element in att_key_num_nodes and att_value_num_nodes
        # is the dimension of the key and the value. In multi-head attention, they are extended n times.
        n_heads = params.att_num_heads
        if params.att_split_value:
            assert shape_list(value_features)[2] % n_heads == 0, "The dim of the value must be divided by the num of heads."
        if params.att_split_key:
            assert shape_list(key_features)[2] % n_heads == 0

        # # TODO: Debug
        # endpoints["att_key"] = key_features
        # endpoints["att_value"] = value_features

        # Split the value and key.
        if params.att_split_value:
            value_features = split_heads(value_features, n_heads)
        else:
            value_features = tf.expand_dims(value_features, axis=1)

        if params.att_split_key:
            key_features = split_heads(key_features, n_heads)
        else:
            key_features = tf.expand_dims(key_features, axis=1)

        val_shape = shape_list(value_features)
        if not params.att_split_value:
            val_shape[1] = n_heads
        key_shape = shape_list(key_features)

        tf.logging.info(
            "Attention:\n"
            "  The dim of the value: %d, the dim of the key: %d\n"
            "  The layer has %d heads, resulting in the dim of value/key each head %d/%d.\n"
            "  With weighted mean and stddev, the attention layer results in output with dim %d."
            % (val_shape[1] * val_shape[-1], key_shape[1] * key_shape[-1],
               n_heads, val_shape[-1], key_shape[-1], val_shape[1] * val_shape[-1] * 2))

        if "att_query_init" in params.dict and params.att_query_init == "xavier_init":
            tf.logging.info("Init query with xavier initializer")
            # Version 2: the initializer is the same as the layer
            query = tf.get_variable("query", [n_heads, key_shape[-1]], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        else:
            # Initialize query thus the weight for each time step is equal at the beginning.
            # Version 1: consistent with BUT implementation
            query = tf.get_variable("query", [n_heads, key_shape[-1]], dtype=tf.float32,
                                    initializer=tf.initializers.truncated_normal(stddev=0.1))

        # # TODO: Debug
        # endpoints["att_query"] = query

        if not params.att_split_key:
            query_time_key = tf.einsum('bmld, hd->blh', key_features, query, name="query_time_key")
        else:
            query_time_key = tf.einsum('bhld, hd->blh', key_features, query, name="query_time_key")

        if params.att_use_scale:
            query_time_key = query_time_key * tf.rsqrt(tf.to_float(key_shape[-1]))

        # weights is [b, h, l]
        weights = tf.nn.softmax(tf.transpose(query_time_key, [0, 2, 1]), name="weights")
        endpoints["attention_weights"] = weights
        # # TODO: Debug
        # endpoints["query_time_key"] = query_time_key

        if params.att_split_value:
            att_mean = tf.einsum('bhld,bhl->bhd', value_features, weights, name="att_mean")
            att_stddev = tf.einsum('bhld,bhl->bhd',
                                   tf.squared_difference(value_features, tf.expand_dims(att_mean, axis=2)), weights,
                                   name="att_stddev")
        else:
            att_mean = tf.einsum('bmld,bhl->bhd', value_features, weights, name="att_mean")
            att_stddev = tf.einsum('bhld,bhl->bhd',
                                   tf.squared_difference(value_features, tf.expand_dims(att_mean, axis=2)), weights,
                                   name="att_stddev")

        att_mean = combine_last_two_dimensions(att_mean)
        att_stddev = combine_last_two_dimensions(att_stddev)
        mask = tf.to_float(tf.less_equal(att_stddev, VAR2STD_EPSILON))
        att_stddev = (1.0 - mask) * att_stddev + mask * VAR2STD_EPSILON
        att_stddev = tf.sqrt(att_stddev)
        att = tf.concat([att_mean, att_stddev], axis=1, name="concat")

        endpoints["att_output_before_nonlinear"] = att

        if params.att_apply_nonlinear:
            att = tf.layers.batch_normalization(att,
                                                momentum=params.batchnorm_momentum,
                                                training=is_training,
                                                name="att_post_bn")
            endpoints["att_post_bn"] = att
            att = relu(att, name='att_post_relu')
            endpoints["att_post_relu"] = att

        # Penalty term when multi-head attention is used.
        if params.att_penalty_term > 0:
            penalty = tf.einsum('ijk,ikl->ijl', weights, tf.transpose(weights, [0, 2, 1])) - tf.eye(n_heads, batch_shape=[val_shape[0]])
            # Normalize using the batch size
            penalty = tf.reduce_sum(tf.square(penalty)) / tf.to_float(val_shape[0])
            penalty = params.att_penalty_term * penalty
            tf.add_to_collection("PENALTY", penalty)
            tf.summary.scalar("attention_penalty", penalty)

    return att


