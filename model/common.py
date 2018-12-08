import tensorflow as tf


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in xrange(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def prelu(x, name="prelu", shared=True):
    """Parametric ReLU

    Args:
        x: the input tensor.
        name: the name of this operation.
        shared: use a shared alpha for all channels.
    """
    alpha_size = 1 if shared else x.get_shape()[-1]
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', alpha_size,
                               initializer=tf.constant_initializer(0.01),
                               dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - abs(x)) * 0.5
    return pos + neg


def l2_normalize(x):
    """Normalize the last dimension vector of the input matrix"""
    l2 = tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True)
    mask = tf.to_float(tf.less(l2, 1e-16))
    norm = tf.sqrt(l2 + mask * 1e-16)
    return x / norm


def pairwise_euc_distances(embeddings, squared=False):
    """Compute the 2D matrix of squared euclidean distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: || x1 - x2 ||^2 or || x1 - x2 ||
    :return: pairwise_square_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances
