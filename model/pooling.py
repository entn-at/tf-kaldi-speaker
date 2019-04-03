import tensorflow as tf
from model.common import shape_list, dense_relu, dense_tanh, split_heads, combine_last_two_dimensions
import sys


VAR2STD_EPSILON = 1e-12


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
    Note that the key should be the same length with the value, i.e. no convnet is applied after the key layer, or
    some trimming strategy should be applied before the weighted sum. (Refer to linguistic_attention)

    Args:
        features: A tensor with shape [batch, length, dim].
        aux_features: Auxiliary input features with shape [batch, length, dim].
        endpoints: Outputs of different parts of the network. Useful when doing attention.
        params: Parameters for self-attention.
            params.self_att_key_input: Use endpoints[params.self_att_key_input] to compute the key.
            params.self_att_key_num_nodes: The network to compute the key.
            params.self_att_value_num_nodes: The network to compute the value.
            params.self_att_num_heads: The number of heads in multi-head attention.
            params.self_att_penalty_term: The coefficient of the penalty term.
            The final dimension of the key and the value is decided by self_att_key_num_nodes and self_att_value_num_nodes.
            If multi-head attention is used, the value will be split first (the key remains the original dim).
        is_training: Used in BN.
    :return:
        Attention result. Also in the statistic format [weighted_mean, weighted_stddev]
    """
    assert "self_att_key_input" in params.dict
    assert "self_att_key_num_nodes" in params.dict
    assert "self_att_value_num_nodes" in params.dict
    assert "self_att_num_heads" in params.dict
    assert "self_att_penalty_term" in params.dict

    with tf.variable_scope("attention"):
        value_features = features
        key_features = endpoints[params.self_att_key_input]

        if len(params.self_att_key_num_nodes) != 0:
            # According to "A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING",
            # the last layer of the key network is `affine + tanh`.
            if len(params.self_att_key_num_nodes) > 1:
                for index, node in enumerate(params.self_att_key_num_nodes[:-1]):
                    key_features = dense_relu(key_features, node, endpoints, params, is_training, name=("att_key%d" % index))
            key_features = dense_tanh(key_features, params.self_att_key_num_nodes[-1], endpoints, params, is_training,
                                      name=("att_key%d" % (len(params.self_att_key_num_nodes) - 1)))

        if len(params.self_att_value_num_nodes) != 0:
            tf.logging.info("Note: Add network to process the value input %s" % value_features.name)
            for index, node in enumerate(params.self_att_value_num_nodes):
                value_features = dense_relu(value_features, node, endpoints, params, is_training, name=("att_value%d" % index))

        # The last element in self_att_key_num_nodes and self_att_value_num_nodes
        # is the dimension of the key and the value. In multi-head attention, they are extended n times.
        n_heads = params.self_att_num_heads
        assert shape_list(value_features)[2] % n_heads == 0, "The dim of the value must be divided by the num of heads."

        # Split the value. The key can use the entire key vector (without splitting).
        value_features = split_heads(value_features, n_heads)
        val_shape = shape_list(value_features)
        key_shape = shape_list(key_features)

        tf.logging.info(
            "Attention:\n"
            "  The dim of the value: %d, the dim of the key: %d\n"
            "  The layer has %d heads, resulting in the dim of value of each head %d.\n"
            "  With weighted mean and stddev, the attention layer results in output with dim %d."
            % (val_shape[1] * val_shape[-1], key_shape[-1], n_heads, val_shape[-1], val_shape[1] * val_shape[-1] * 2))

        # Initialize query thus the weight for each time step is equal at the beginning.
        query = tf.get_variable("query", [n_heads, key_shape[-1]], dtype=tf.float32,
                                initializer=tf.initializers.truncated_normal(stddev=0.1))

        query_time_key = tf.einsum('ijl,kl->ijk', key_features, query, name="query_time_key")
        weights = tf.nn.softmax(tf.transpose(query_time_key, [0, 2, 1]), name="weights")

        att_mean = tf.einsum('bnld,bnl->bnd', value_features, weights, name="att_mean")
        att_stddev = tf.einsum('bnld,bnl->bnd',
                               tf.squared_difference(value_features, tf.expand_dims(att_mean, axis=2)), weights,
                               name="att_stddev")

        att_mean = combine_last_two_dimensions(att_mean)
        att_stddev = combine_last_two_dimensions(att_stddev)

        mask = tf.to_float(tf.less_equal(att_stddev, VAR2STD_EPSILON))
        att_stddev = (1.0 - mask) * att_stddev + mask * VAR2STD_EPSILON
        att_stddev = tf.sqrt(att_stddev)

        att = tf.concat([att_mean, att_stddev], 1, name="concat")
        endpoints["attention_weights"] = weights

        # Penalty term
        penalty = tf.einsum('ijk,ikl->ijl', weights, tf.transpose(weights, [0, 2, 1])) - tf.eye(n_heads, batch_shape=[val_shape[0]])
        # Normalize using the batch size
        penalty = tf.reduce_sum(tf.square(penalty)) / tf.to_float(val_shape[0])
        tf.add_to_collection("PENALTY", params.self_att_penalty_term * penalty)
        tf.summary.scalar("attention_penalty", params.self_att_penalty_term * penalty)

        # # Debug
        # # Comment lines when running the code
        # endpoints["att_query"] = query
        # endpoints["att_key"] = key_features
        # endpoints["att_value"] = value_features
    return att


def aux_attention(features, aux_features, endpoints, params, is_training=None):
    """Attention using auxiliary features.

    The attention layer has a minor problem that the length of the key may be different with the length of the value,
    due to the convnet. The key usually has the original feature length while the length of the value is shorter.
    We always using the fully-connected layer in the key network, so the length remains the same.
    A workaround is to use the center of the key to make length of the key and the value the same.

    Note: When auxiliary key is used, the hypothesis is that the length of this auxiliary feature is the same with the value.

    Args:
        features: A tensor with shape [batch, length, dim].
        aux_features: A dict.
        aux_featuers["aux_feat_name"]: The length is LONGER than features!!!
                                    The features is processed by convnet thus the length becomes shorter.
        TODO: How to trim the auxiliary features? Align left or center?
        endpoints: Outputs of different parts of the network.
        params: Parameters for self-attention.
            params.att_aux_name: The name of the auxiliary features.
            params.att_aux_key_input: Additional key input except for the auxiliary features.
                                      If None then only the auxiliary features are used.
            params.att_key_num_nodes: The network to compute the key.
            params.att_value_num_nodes: The network to compute the value.
            params.att_num_heads: The number of heads in multi-head attention.
            params.att_penalty_term: The coefficient of the penalty term.
            The final dimension of the key and the value is decided by self_att_key_num_nodes and self_att_value_num_nodes.
            If multi-head attention is used, the value will be split first (the key remains the original dim).
        is_training: Used in BN.
    :return:
    """
    assert "att_aux_name" in params.dict
    assert "att_key_input" in params.dict
    assert "att_key_num_nodes" in params.dict
    assert "att_value_num_nodes" in params.dict
    assert "att_num_heads" in params.dict
    assert "att_penalty_term" in params.dict

    with tf.variable_scope("attention"):
        value_features = features
        for aux_name in params.att_aux_name:
            if aux_name not in aux_features:
                sys.exit("The aux features %s is not in aux_features." % aux_name)

        key_features = []
        for aux_name in params.att_aux_name:
            # Center trimming. Use the center of the key to match the length of the value.
            trim_length = (shape_list(aux_features[aux_name])[1] - shape_list(value_features)[1]) / 2
            # This requires the total kernel size is a odd number.
            key_features.append(aux_features[aux_name][:, trim_length:-trim_length, :])

            # # TODO: If the length of the key and the value is the same, the next line is useful.
            # # But the above line looks more neat (What...).
            # key_features = tf.cond(tf.equal(trim_length, 0),
            #                        lambda: aux_features[aux_name],
            #                        lambda: aux_features[aux_name][:, trim_length:-trim_length, :])

        tf.logging.info("Attention using auxiliary features:")
        if params.att_key_input is not None:
            if params.att_key_input not in endpoints:
                sys.exit(
                    "You specify the appended key %s, but I cannot find it in the endpoints." % params.att_key_input)
            tf.logging.info("Append %s to the auxiliary features" % params.att_key_input)
            key_features.append(endpoints[params.att_key_input])

        # Concatenate all the features to the key.
        key_features = tf.concat(key_features, axis=-1, name="key_features")

        if len(params.att_key_num_nodes) != 0:
            # According to "A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING",
            # the last layer of the key network is `affine + tanh`.
            if len(params.att_key_num_nodes) > 1:
                for index, node in enumerate(params.att_key_num_nodes[:-1]):
                    key_features = dense_relu(key_features, node, endpoints, params, is_training, name=("att_key%d" % index))
            key_features = dense_tanh(key_features, params.att_key_num_nodes[-1], endpoints, params, is_training,
                                      name=("att_key%d" % (len(params.att_key_num_nodes) - 1)))

        if len(params.att_value_num_nodes) != 0:
            tf.logging.info("Note: Add network to process the value input %s" % value_features.name)
            for index, node in enumerate(params.att_value_num_nodes):
                value_features = dense_relu(value_features, node, endpoints, params, is_training, name=("att_value%d" % index))

        # The last element in self_att_key_num_nodes and self_att_value_num_nodes
        # is the dimension of the key and the value. In multi-head attention, they are extended n times.
        n_heads = params.att_num_heads
        assert shape_list(value_features)[2] % n_heads == 0, "The dim of the value must be divided by the num of heads."

        # Split the value. The key can use the entire vector.
        value_features = split_heads(value_features, n_heads)
        val_shape = shape_list(value_features)
        key_shape = shape_list(key_features)

        tf.logging.info(
            "  The dim of the value: %d, the dim of the key: %d\n"
            "  The layer has %d heads, resulting in the dim of value of each head %d.\n"
            "  With weighted mean and stddev, the attention layer results in output with dim %d."
            % (val_shape[1] * val_shape[-1], key_shape[-1], n_heads, val_shape[-1], val_shape[1] * val_shape[-1] * 2))

        # Initialize query thus the weight for each time step is equal at the beginning.
        query = tf.get_variable("query", [n_heads, key_shape[-1]], dtype=tf.float32,
                                initializer=tf.initializers.truncated_normal(stddev=0.1))

        query_time_key = tf.einsum('ijl,kl->ijk', key_features, query, name="query_time_key")
        weights = tf.nn.softmax(tf.transpose(query_time_key, [0, 2, 1]), name="weights")

        att_mean = tf.einsum('bnld,bnl->bnd', value_features, weights, name="att_mean")
        att_stddev = tf.einsum('bnld,bnl->bnd',
                               tf.squared_difference(value_features, tf.expand_dims(att_mean, axis=2)), weights,
                               name="att_stddev")

        att_mean = combine_last_two_dimensions(att_mean)
        att_stddev = combine_last_two_dimensions(att_stddev)

        mask = tf.to_float(tf.less_equal(att_stddev, VAR2STD_EPSILON))
        att_stddev = (1.0 - mask) * att_stddev + mask * VAR2STD_EPSILON
        att_stddev = tf.sqrt(att_stddev)

        att = tf.concat([att_mean, att_stddev], 1, name="concat")
        endpoints["attention_weights"] = weights

        # Penalty term
        penalty = tf.einsum('ijk,ikl->ijl', weights, tf.transpose(weights, [0, 2, 1])) - tf.eye(n_heads, batch_shape=[val_shape[0]])
        penalty = tf.reduce_sum(tf.square(penalty)) / tf.to_float(val_shape[0])
        tf.add_to_collection("PENALTY", params.att_penalty_term * penalty)
        tf.summary.scalar("attention_penalty", params.att_penalty_term * penalty)

        # # Debug
        # # Comment lines when running the code
        # endpoints["att_query"] = query
        # endpoints["att_key"] = key_features
        # endpoints["att_value"] = value_features
    return att


if __name__ == "__main__":
    num_labels = 10
    num_data = 100
    num_length = 100
    num_dim = 1500
    features = tf.placeholder(tf.float32, shape=[None, None, num_dim], name="features")
    aux_features = tf.placeholder(tf.float32, shape=[None, None, 100], name="aux_features")
    linguistic_features = tf.placeholder(tf.float32, shape=[None, None, 500], name="linguistic_features")
    linguistic_features_all = {"linguistic": linguistic_features}
    from collections import OrderedDict
    endpoints = OrderedDict()

    from misc.utils import ParamsPlain

    # # Self-attention (key transform)
    # params = ParamsPlain()
    # params.dict["self_att_key_input"] = "key"
    # params.dict["self_att_key_num_nodes"] = [1500, 1500]
    # params.dict["self_att_value_num_nodes"] = []
    # params.dict["self_att_num_heads"] = 10
    # params.dict["self_att_penalty_term"] = 1
    # params.dict["weight_l2_regularizer"] = 1e-2
    # params.dict["batchnorm_momentum"] = 0.99
    #
    # endpoints["key"] = features
    # self_att = self_attention(features, aux_features, endpoints, params, is_training=True)
    # penalty_loss = tf.reduce_sum(tf.get_collection("PENALTY"))
    # grads = tf.gradients(self_att, features)
    # grads_penalty = tf.gradients(penalty_loss, features)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     import numpy as np
    #     features_val = np.random.rand(num_data, num_length, num_dim).astype(np.float32)
    #     features_val[0, :, :] = 1e-8 * features_val[0, :, :]
    #     features_val[1, :, :] = 0
    #     features_val[2, :, :] = 100 * features_val[2, :, :]
    #     features_val[3, :, :] = 100
    #     self_att_val, penalty_loss_val, grads_val, grads_penalty_val, endpoints_val = sess.run([self_att, penalty_loss, grads, grads_penalty, endpoints], feed_dict={features: features_val})
    #     key = endpoints_val["att_key"]
    #     value = endpoints_val["att_value"]
    #     query = endpoints_val["att_query"]
    #
    #     from model.test_utils import compute_self_attention
    #     self_att_np, penalty_loss_np = compute_self_attention(value, key, query, params)
    #
    #     assert np.allclose(np.sum(self_att_val), np.sum(self_att_np))
    #     assert np.allclose(penalty_loss_val, penalty_loss_np)
    #
    #     assert not np.any(np.isnan(grads_val)), "Gradient should not be nan"
    #     assert not np.any(np.isnan(grads_penalty_val)), "Gradient should not be nan"

    # # Linguistic attention (key transform)
    # params = ParamsPlain()
    # params.dict["att_aux_key_input"] = "key"
    # params.dict["att_key_num_nodes"] = [1500, 1500]
    # params.dict["att_value_num_nodes"] = []
    # params.dict["att_num_heads"] = 1
    # params.dict["att_penalty_term"] = 1
    # params.dict["weight_l2_regularizer"] = 1e-2
    # params.dict["batchnorm_momentum"] = 0.99
    # 
    # endpoints["key"] = aux_features
    # att = linguistic_attention(features, linguistic_features_all, endpoints, params, is_training=True)
    # penalty_loss = tf.reduce_sum(tf.get_collection("PENALTY"))
    # grads = tf.gradients(att, features)
    # grads_penalty = tf.gradients(penalty_loss, linguistic_features_all["linguistic"])
    # 
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     import numpy as np
    # 
    #     features_val = np.random.rand(num_data, num_length, num_dim).astype(np.float32)
    #     features_val[0, :, :] = 1e-8 * features_val[0, :, :]
    #     features_val[1, :, :] = 0
    #     features_val[2, :, :] = 100 * features_val[2, :, :]
    #     features_val[3, :, :] = 100
    # 
    #     aux_features_val = np.random.rand(num_data, num_length, 100).astype(np.float32)
    #     linguistic_features_val = np.random.rand(num_data, num_length, 500).astype(np.float32)
    # 
    #     att_val, penalty_loss_val, grads_val, grads_penalty_val, endpoints_val = sess.run(
    #         [att, penalty_loss, grads, grads_penalty, endpoints], feed_dict={features: features_val,
    #                                                                          aux_features: aux_features_val,
    #                                                                          linguistic_features: linguistic_features_val})
    #     key = endpoints_val["att_key"]
    #     value = endpoints_val["att_value"]
    #     query = endpoints_val["att_query"]
    # 
    #     from model.test_utils import compute_attention
    #     att_np, penalty_loss_np = compute_attention(value, key, query, params)
    # 
    #     assert np.allclose(np.sum(att_val), np.sum(att_np))
    #     assert np.allclose(penalty_loss_val, penalty_loss_np)
    # 
    #     assert not np.any(np.isnan(grads_val)), "Gradient should not be nan"
    #     assert not np.any(np.isnan(grads_penalty_val)), "Gradient should not be nan"

    # Self-attention
    params = ParamsPlain()
    params.dict["self_att_key_input"] = "key"
    params.dict["self_att_key_num_nodes"] = []
    params.dict["self_att_value_num_nodes"] = []
    params.dict["self_att_num_heads"] = 10
    params.dict["self_att_penalty_term"] = 1
    params.dict["weight_l2_regularizer"] = 1e-2
    params.dict["batchnorm_momentum"] = 0.99

    endpoints["key"] = features
    self_att = self_attention(features, aux_features, endpoints, params, is_training=True)
    penalty_loss = tf.reduce_sum(tf.get_collection("PENALTY"))
    grads = tf.gradients(self_att, features)
    grads_penalty = tf.gradients(penalty_loss, features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import numpy as np
        features_val = np.random.rand(num_data, num_length, num_dim).astype(np.float32)
        features_val[0, :, :] = 1e-8 * features_val[0, :, :]
        features_val[1, :, :] = 0
        features_val[2, :, :] = 100 * features_val[2, :, :]
        features_val[3, :, :] = 100
        self_att_val, penalty_loss_val, grads_val, grads_penalty_val, endpoints_val = sess.run([self_att, penalty_loss, grads, grads_penalty, endpoints], feed_dict={features: features_val})
        query = endpoints_val["att_query"]
        value = np.reshape(features_val, [features_val.shape[0], features_val.shape[1], params.self_att_num_heads,
                                          features_val.shape[2] / params.self_att_num_heads])
        value = np.transpose(value, [0,2,1,3])
        key = features_val

        from model.test_utils import compute_self_attention
        self_att_np, penalty_loss_np = compute_self_attention(value, key, query, params)

        assert np.allclose(np.sum(self_att_val), np.sum(self_att_np))
        assert np.allclose(penalty_loss_val, penalty_loss_np)

        assert not np.any(np.isnan(grads_val)), "Gradient should not be nan"
        assert not np.any(np.isnan(grads_penalty_val)), "Gradient should not be nan"

    # # Linguistic attention
    # params = ParamsPlain()
    # params.dict["att_aux_key_input"] = "key"
    # params.dict["att_key_num_nodes"] = []
    # params.dict["att_value_num_nodes"] = []
    # params.dict["att_num_heads"] = 1
    # params.dict["att_penalty_term"] = 1
    # params.dict["weight_l2_regularizer"] = 1e-2
    # params.dict["batchnorm_momentum"] = 0.99
    #
    # endpoints["key"] = aux_features
    # att = linguistic_attention(features, linguistic_features_all, endpoints, params, is_training=True)
    # penalty_loss = tf.reduce_sum(tf.get_collection("PENALTY"))
    # grads = tf.gradients(att, features)
    # grads_penalty = tf.gradients(penalty_loss, linguistic_features_all["linguistic"])
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     import numpy as np
    #
    #     features_val = np.random.rand(num_data, num_length-4, num_dim).astype(np.float32)
    #     features_val[0, :, :] = 1e-8 * features_val[0, :, :]
    #     features_val[1, :, :] = 0
    #     features_val[2, :, :] = 100 * features_val[2, :, :]
    #     features_val[3, :, :] = 100
    #
    #     aux_features_val = np.random.rand(num_data, num_length-4, 100).astype(np.float32)
    #     linguistic_features_val = np.random.rand(num_data, num_length, 500).astype(np.float32)
    #
    #     att_val, penalty_loss_val, grads_val, grads_penalty_val, endpoints_val = sess.run(
    #         [att, penalty_loss, grads, grads_penalty, endpoints], feed_dict={features: features_val,
    #                                                                          aux_features: aux_features_val,
    #                                                                          linguistic_features: linguistic_features_val})
    #     query = endpoints_val["att_query"]
    #     value = np.reshape(features_val, [features_val.shape[0], features_val.shape[1], params.att_num_heads, features_val.shape[2]/params.att_num_heads])
    #     value = np.transpose(value, [0,2,1,3])
    #     key = np.concatenate([linguistic_features_val[:,2:-2,:], aux_features_val], axis=-1)
    #
    #     from model.test_utils import compute_attention
    #     att_np, penalty_loss_np = compute_attention(value, key, query, params)
    #
    #     assert np.allclose(np.sum(att_val), np.sum(att_np))
    #     assert np.allclose(penalty_loss_val, penalty_loss_np)
    #
    #     assert not np.any(np.isnan(grads_val)), "Gradient should not be nan"
    #     assert not np.any(np.isnan(grads_penalty_val)), "Gradient should not be nan"
