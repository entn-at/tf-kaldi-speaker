import tensorflow as tf


def statistics_pooling(features):
    """Statistics pooling
    Note that we need to take care of the zeros in the variance since the sqrt on 0 will lead to NaN.

    Args:
        features: A tensor with shape [batch, length, dim]
    :return:
        Statistics pooling result [mean, stddev].
    """
    mean = tf.reduce_mean(features, axis=1, keep_dims=True, name="mean")
    variance = tf.reduce_mean(tf.squared_difference(features, mean), axis=1, keep_dims=True, name="variance")
    mean = tf.squeeze(mean, 1)
    variance = tf.squeeze(variance, 1)

    # Because the gradient of sqrt is infinite when variance == 0.0
    mask = tf.to_float(tf.less_equal(variance, 0.0))
    variance = variance + mask * 1e-16
    stddev = tf.sqrt(variance)
    stddev = stddev * (1.0 - mask)

    stat_pooling = tf.concat([mean, stddev], 1, name="concat")

    return stat_pooling
