import numpy as np
from six.moves import range


def compute_cos(x1, x2):
    """Compute cosine similarity between x1 and x2"""
    return np.dot(x1, np.transpose(x2)) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-16)


def sigmoid(x):
    """Sigmoid transform."""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.maximum(e_x.sum(axis=-1, keepdims=True), 1e-16)


def compute_ge2e_loss(embeddings, labels, w, b, ge2e_type):
    """Compute generalized end-to-end loss. This is simply used to check the tf implementation in loss.py.

    Args:
        embeddings: The input features without l2 normalization.
        labels: The labels to compute the loss.
        w: The initial w value.
        b: The initial b value.
        ge2e_type: "softmax" or "contrastive"
    :return: The loss value.
    """
    embeddings /= np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True) + 1e-16)
    class_index = []
    label2class = {}
    for l in labels:
        if l not in label2class:
            label2class[l] = len(label2class.keys())
        class_index.append(label2class[l])
    n_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]
    n_classes = len(label2class.keys())
    sim = np.zeros((n_samples, n_classes))
    centers = np.zeros((n_classes, n_dim))
    for i in range(n_classes):
        n_class_samples = 0
        for j in range(n_samples):
            if class_index[j] != i:
                continue
            centers[i, :] += embeddings[j, :]
            n_class_samples += 1
        centers[i, :] /= n_class_samples
        centers /= np.sqrt(np.sum(centers ** 2, axis=1, keepdims=True) + 1e-16)

    for i in range(n_samples):
        for j in range(n_classes):
            if class_index[i] == j:
                center_exclusive = np.zeros((1, n_dim))
                n_exclusive_samples = 0
                for k in range(n_samples):
                    if class_index[k] != j or k == i:
                        continue
                    center_exclusive += embeddings[k, :]
                    n_exclusive_samples += 1
                center_exclusive /= np.sqrt(np.sum(center_exclusive ** 2, axis=1, keepdims=True) + 1e-16)
                sim[i, j] = w * compute_cos(embeddings[i, :], center_exclusive / (n_exclusive_samples + 1e-16)) + b
            else:
                sim[i, j] = w * compute_cos(embeddings[i, :], centers[j, :]) + b

    n_samples, n_classes = sim.shape
    loss = 0

    if ge2e_type == "softmax":
        s = softmax(sim)
        for i in range(n_samples):
            loss -= np.log(s[i, class_index[i]] + 1e-16)
            # loss -= sim[i, class_index[i]] - np.log(np.sum(np.exp(sim[i, :])) + 1e-16)
    else:
        for i in range(n_samples):
            other = [0]
            for j in range(n_classes):
                if class_index[i] == j:
                    continue
                other.append(sigmoid(sim[i, j]))
            other = sorted(other)
            loss += 1 - sigmoid(sim[i, class_index[i]]) + other[-1]
    return loss / n_samples


def pairwise_euc_distances_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.
    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: The L2 distance or square root of the distance.
    Returns:
        square_pairwise_distances:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    square_upper_tri_pdists = upper_tri_pdists ** 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    square_pairwise_distances = np.zeros((num_data, num_data))
    square_pairwise_distances[np.triu_indices(num_data, 1)] = square_upper_tri_pdists

    # Make symmetrical.
    if squared:
        distances = square_pairwise_distances + square_pairwise_distances.T - np.diag(
            square_pairwise_distances.diagonal())
    else:
        distances = pairwise_distances + pairwise_distances.T - np.diag(
                pairwise_distances.diagonal())
    return distances


def compute_triplet_loss(embeddings, labels, margin, squared):
    """Compute the triplet loss. This is used to check the tf implementation in loss.py

    Args:
        embeddings: The input features.
        labels: The labels.
        margin: The margin in triplet loss.
        squared: The distance is squared or not.
    :return: The triplet loss
    """
    embeddings /= np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))
    num_data = embeddings.shape[0]
    distances = pairwise_euc_distances_np(embeddings, squared)
    loss_np = 0.0
    num_positives_np = 0
    for i in range(num_data):
        for j in range(num_data):
            d_xy = distances[i, j]
            semi_hard_dist = []
            all_dist = []

            for k in range(num_data):
                if labels[k] != labels[i]:
                    all_dist.append(distances[i, k])
                    if distances[i, k] > d_xy:
                        semi_hard_dist.append(distances[i, k])

            if len(semi_hard_dist) == 0:
                d_xz = np.amax(all_dist)
            else:
                d_xz = np.amin(semi_hard_dist)

            if labels[i] == labels[j] and i != j:
                loss = np.maximum(0.0, margin + d_xy - d_xz)
                loss_np += loss
                num_positives_np += 1
    return loss_np / num_positives_np


def compute_asoftmax(embeddings, labels, params, w):
    """Compute the angular-softmax loss. This is used to check the tf implementation in loss.py

        Args:
            embeddings: The input features.
            labels: The labels.
            params: some parameters used in asoftmax.
            w: the weight matrix of W
        :return: The angular loss
    """
    n = embeddings.shape[0]
    embeddings_norm = np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True) + 1e-16)
    prod = np.dot(embeddings, w)
    prod /= np.sqrt(np.sum(w ** 2, axis=0, keepdims=True) + 1e-16)
    cosine = prod / embeddings_norm
    if params.feature_norm:
        logits = params.feature_scaling_factor * cosine
    else:
        logits = embeddings_norm * cosine

    if params.asoftmax_m == 1:
        prob = softmax(logits)
        loss = 0
        for i in range(n):
            loss -= np.log(prob[i, labels[i]] + 1e-16)
        return loss / n

    lamb = max(params.asoftmax_lambda_min, params.asoftmax_lambda_base * (1.0 + params.asoftmax_lambda_gamma * params.global_step) ** (-params.asoftmax_lambda_power))
    fa = 1.0 / (1.0 + lamb)
    fs = 1.0 - fa

    cosine = np.minimum(np.maximum(cosine, -1), 1)
    if params.asoftmax_m == 2:
        for i in range(n):
            if cosine[i, labels[i]] > 0:
                k = 0
            else:
                k = 1
            cosine[i, labels[i]] = fa * (((-1) ** k) * (np.cos(2 * np.arccos(cosine[i, labels[i]]))) - 2 * k) + fs * cosine[i, labels[i]]
        if params.feature_norm:
            logits = params.feature_scaling_factor * cosine
        else:
            logits = embeddings_norm * cosine
        prob = softmax(logits)
        loss = 0
        for i in range(n):
            loss -= np.log(prob[i, labels[i]] + 1e-16)
        return loss / n

    assert params.asoftmax_m == 4
    for i in range(n):
        l = np.cos(2 * np.arccos(cosine[i, labels[i]]))
        if cosine[i, labels[i]] > 0 and l > 0:
            k = 0
        elif cosine[i, labels[i]] > 0 and l < 0:
            k = 1
        elif cosine[i, labels[i]] < 0 and l < 0:
            k = 2
        else:
            k = 3
        cosine[i, labels[i]] = fa * (((-1) ** k) * (np.cos(4 * np.arccos(cosine[i, labels[i]]))) - 2 * k) + fs * cosine[i, labels[i]]
    if params.feature_norm:
        logits = params.feature_scaling_factor * cosine
    else:
        logits = embeddings_norm * cosine
    prob = softmax(logits)
    loss = 0
    for i in range(n):
        loss -= np.log(prob[i, labels[i]] + 1e-16)
    return loss / n


def compute_amsoftmax(embeddings, labels, params, w):
    """Compute the additive margin softmax loss. This is used to check the tf implementation in loss.py

        Args:
            embeddings: The input features.
            labels: The labels.
            params: some parameters used in asoftmax.
            w: the weight matrix of W
        :return: The additive margin loss
    """
    n = embeddings.shape[0]
    embeddings_norm = np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True) + 1e-16)
    prod = np.dot(embeddings, w)
    prod /= np.sqrt(np.sum(w ** 2, axis=0, keepdims=True) + 1e-16)
    cos_theta = prod / embeddings_norm
    cos_theta = np.minimum(np.maximum(cos_theta, -1), 1)

    if params.feature_norm:
        logits_org = params.feature_scaling_factor * cos_theta
    else:
        logits_org = embeddings_norm * cos_theta

    for i in range(n):
        cos_theta[i, labels[i]] -= params.amsoftmax_m

    if params.feature_norm:
        logits = params.feature_scaling_factor * cos_theta
    else:
        logits = embeddings_norm * cos_theta

    lamb = max(params.amsoftmax_lambda_min,
               params.amsoftmax_lambda_base * (1.0 + params.amsoftmax_lambda_gamma * params.global_step) ** (
                   -params.amsoftmax_lambda_power))
    fa = 1.0 / (1.0 + lamb)
    fs = 1.0 - fa
    logits = fs * logits_org + fa * logits

    prob = softmax(logits)
    loss = 0
    for i in range(n):
        loss -= np.log(prob[i, labels[i]]+1e-16)
    return loss / n


def compute_arcsoftmax(embeddings, labels, params, w):
    """Compute the additive angular margin softmax loss. This is used to check the tf implementation in loss.py

        Args:
            embeddings: The input features.
            labels: The labels.
            params: some parameters used in asoftmax.
            w: the weight matrix of W
        :return: The additive angular margin loss
    """
    n = embeddings.shape[0]
    embeddings_norm = np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True) + 1e-16)
    prod = np.dot(embeddings, w)
    prod /= np.sqrt(np.sum(w ** 2, axis=0, keepdims=True) + 1e-16)
    cos_theta = prod / embeddings_norm
    cos_theta = np.minimum(np.maximum(cos_theta, -1), 1)

    if params.feature_norm:
        logits_org = params.feature_scaling_factor * cos_theta
    else:
        logits_org = embeddings_norm * cos_theta

    for i in range(n):
        angle = np.arccos(cos_theta[i, labels[i]]) + params.arcsoftmax_m
        if angle > np.pi:
            cos_theta[i, labels[i]] = -np.cos(angle) - 2
        else:
            cos_theta[i, labels[i]] = np.cos(angle)

    if params.feature_norm:
        logits = params.feature_scaling_factor * cos_theta
    else:
        logits = embeddings_norm * cos_theta

    lamb = max(params.arcsoftmax_lambda_min,
               params.arcsoftmax_lambda_base * (1.0 + params.arcsoftmax_lambda_gamma * params.global_step) ** (
                   -params.arcsoftmax_lambda_power))
    fa = 1.0 / (1.0 + lamb)
    fs = 1.0 - fa
    logits = fs * logits_org + fa * logits

    prob = softmax(logits)
    loss = 0
    for i in range(n):
        loss -= np.log(prob[i, labels[i]] + 1e-16)
    return loss / n


def compute_self_attention(value, key, query, params):
    """Compute the output of the self attention layer.

    Args:
        value/key/query: The value, key and query of the attention.
    :return:
        att: The output of the attention layer.
        penalty_term: The penalty term of the attention layer.
    """
    batch, length, value_dim = value.shape
    key_dim = key.shape[-1]
    n_heads = query.shape[0]

    if params.att_split_value:
        value = np.transpose(np.reshape(value, [batch, length, n_heads, value_dim/n_heads]), [0,2,1,3])

    if params.att_split_key:
        key = np.transpose(np.reshape(key, [batch, length, n_heads, key_dim/n_heads]), [0,2,1,3])
    else:
        key = np.expand_dims(key, axis=1)

    if params.att_use_scale:
        scale = 1.0 / np.sqrt(key.shape[-1])
    else:
        scale = 1.0

    query_time_key = np.zeros((batch, n_heads, length))
    for i in range(batch):
        for j in range(n_heads):
            for k in range(length):
                if params.att_split_key:
                    query_time_key[i, j, k] = np.sum(key[i, j, k, :] * query[j, :]) * scale
                else:
                    query_time_key[i, j, k] = np.sum(key[i, 0, k, :] * query[j, :]) * scale

    weights = np.zeros((batch, n_heads, length))
    for i in range(batch):
        for j in range(n_heads):
            weights[i, j, :] = softmax(query_time_key[i, j, :])

    p = np.zeros((batch, n_heads, n_heads))
    for i in range(batch):
        p[i, :, :] = np.dot(weights[i, :, :], np.transpose(weights[i, :, :])) - np.eye(n_heads)
    penalty = params.att_penalty_term * np.sum(np.square(p)) / batch

    if params.att_split_value:
        att_mean = np.zeros((batch, n_heads, value_dim/n_heads))
        att_stddev = np.zeros((batch, n_heads, value_dim/n_heads))
        for i in range(batch):
            for j in range(n_heads):
                att_mean[i, j, :] = np.dot(weights[i, j, :][np.newaxis, :], value[i, j, :, :])
                tmp = np.dot(weights[i, j, :][np.newaxis, :], (value[i, j, :, :] - att_mean[i, j, :][np.newaxis, :]) ** 2)
                tmp[tmp<1e-12] = 1e-12
                att_stddev[i, j, :] = np.sqrt(tmp)

        att_mean = np.reshape(att_mean, [batch, value_dim])
        att_stddev = np.reshape(att_stddev, [batch, value_dim])
    else:
        att_mean = np.zeros((batch, n_heads, value_dim))
        att_stddev = np.zeros((batch, n_heads, value_dim))
        for i in range(batch):
            for j in range(n_heads):
                att_mean[i, j, :] = np.dot(weights[i, j, :][np.newaxis, :], value[i, :, :])
                tmp = np.dot(weights[i, j, :][np.newaxis, :], (value[i, :, :] - att_mean[i, j, :][np.newaxis, :]) ** 2)
                tmp[tmp < 1e-12] = 1e-12
                att_stddev[i, j, :] = np.sqrt(tmp)

        att_mean = np.reshape(att_mean, [batch, value_dim*n_heads])
        att_stddev = np.reshape(att_stddev, [batch, value_dim*n_heads])

    att = np.concatenate([att_mean, att_stddev], axis=1)
    return att, penalty


def compute_attention(value, key, query, params):
    """Compute the output of a general attention layer.

    Args:
        value/key/query: The value, key and query of the attention.
    :return:
        att: The output of the attention layer.
        penalty_term: The penalty term of the attention layer.
    """
    batch, length, dim = key.shape
    dim = value.shape[-1]
    n_heads = query.shape[0]

    query_time_key = np.zeros((batch, n_heads, length))
    for i in range(batch):
        for j in range(n_heads):
            for k in range(length):
                query_time_key[i, j, k] = np.sum(key[i, k, :] * query[j, :])

    weights = np.zeros((batch, n_heads, length))
    for i in range(batch):
        for j in range(n_heads):
            weights[i, j, :] = softmax(query_time_key[i, j, :])

    p = np.zeros((batch, n_heads, n_heads))
    for i in range(batch):
        p[i, :, :] = np.dot(weights[i, :, :], np.transpose(weights[i, :, :])) - np.eye(n_heads)
    penalty = params.att_penalty_term * np.sum(np.square(p))

    att_mean = np.zeros((batch, n_heads, dim))
    att_stddev = np.zeros((batch, n_heads, dim))

    for i in range(batch):
        for j in range(n_heads):
            att_mean[i, j, :] = np.dot(weights[i, j, :], value[i, j, :, :])
            att_stddev[i, j, :] = np.sqrt(np.dot(weights[i, j, :], (value[i, j, :, :] - att_mean[i, j, :]) ** 2) + 1e-12)
    att_mean = np.reshape(att_mean, [batch, n_heads * dim])
    att_stddev = np.reshape(att_stddev, [batch, n_heads * dim])
    att = np.concatenate([att_mean, att_stddev], axis=1)
    return att, penalty


def compute_ghost_vlad(value, key, centers, params):
    """Compute NetVLAD or GhostVLAD
    """
    # Get the posterior
    post = softmax(key)
    res = np.expand_dims(value, axis=2) - centers[np.newaxis, np.newaxis, :, :]
    res = np.sum(res * np.expand_dims(post, axis=3), axis=1)

    # Remove the ghost centers
    res = res[:, :params.vlad_num_centers, :]

    res /= np.sqrt(np.sum(res ** 2, axis=-1, keepdims=True))
    output = np.reshape(res, [value.shape[0], res.shape[1] * res.shape[2]])

    if params.vlad_use_variance:
        batch_size, length, dim = value.shape
        res_sq = np.zeros((batch_size, length, params.vlad_num_centers + params.vlad_num_ghosts, dim))
        for i in range(batch_size):
            for j in range(length):
                for k in range(params.vlad_num_centers + params.vlad_num_ghosts):
                    res_sq[i, j, k, :] = ((value[i, j, :] - centers[k, :]) ** 2) * post[i, j, k]
        res_sq = np.sum(res_sq, axis=1)
        res_sq = res_sq[:, :params.vlad_num_centers, :]
        res_sq = np.sqrt(res_sq + 1e-16)
        res_sq /= np.sqrt(np.sum(res_sq ** 2, axis=-1, keepdims=True))
        output_sq = np.reshape(res_sq, [value.shape[0], res.shape[1] * res.shape[2]])
        output = np.concatenate([output, output_sq], axis=-1)

    if params.vlad_final_l2_norm:
        output /= (np.sqrt(np.sum(output ** 2, axis=1, keepdims=True)) + 1e-16)
    return output


def compute_phonetic_vlad(value, posteriors, centers, params):
    """Compute PhoneticVLAD
    """
    batch_size, length, dim = value.shape
    num_classes = centers.shape[0]
    assert(num_classes == posteriors.shape[-1])
    res = np.zeros((batch_size, length, num_classes, dim))
    for i in range(batch_size):
        for j in range(length):
            for k in range(num_classes):
                res[i, j, k, :] = value[i, j, :] - centers[k, :]
    res = np.sum(res * np.expand_dims(posteriors, axis=3), axis=1)
    res /= np.sqrt(np.sum(res ** 2, axis=-1, keepdims=True))
    classes = []
    for i in range(num_classes):
        if i not in params.vlad_useless_classes:
            classes.append(i)
    # num_valid = len(classes)
    res = res[:, classes, :]
    output = np.reshape(res, [res.shape[0], res.shape[1] * res.shape[2]])

    if params.vlad_use_variance:
        res_sq = np.zeros((batch_size, length, num_classes, dim))
        for i in range(batch_size):
            for j in range(length):
                for k in range(num_classes):
                    res_sq[i, j, k, :] = ((value[i, j, :] - centers[k, :]) ** 2) * posteriors[i, j, k]
        res_sq = np.sum(res_sq, axis=1)
        res_sq = res_sq[:, classes, :]
        res_sq = np.sqrt(res_sq + 1e-16)
        res_sq /= np.sqrt(np.sum(res_sq ** 2, axis=-1, keepdims=True))
        output_sq = np.reshape(res_sq, [res_sq.shape[0], res_sq.shape[1] * res_sq.shape[2]])
        output = np.concatenate([output, output_sq], axis=-1)
    return output


def compute_ghost_vlad_wocenter(value, key, params):
    """Compute NetVLAD or GhostVLAD without centering.
    """
    # Get the posterior
    post = softmax(key)

    # The zero-order stat
    zero = np.sum(post, axis=1)
    zero[zero<1e-16] = 1e-16

    b, l, d = value.shape
    k = key.shape[2]

    # The first-order stat
    first = np.zeros((b, k, d))
    for i in range(b):
        for m in range(k):
            for j in range(l):
                first[i, m, :] += (post[i, j, m] * value[i, j, :])
    mean = first / np.expand_dims(zero, axis=2)

    # The second-order stat
    second = np.zeros((b, k, d))
    for i in range(b):
        for m in range(k):
            for j in range(l):
                second[i, m, :] += (post[i, j, m] * ((value[i, j, :] - mean[i, m, :]) ** 2))
    var = second / np.expand_dims(zero, axis=2)

    # Remove the ghost centers
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var)

    mean = mean[:, :params.vlad_num_centers, :]
    std = std[:, :params.vlad_num_centers, :]
    output = np.reshape(mean, [value.shape[0], value.shape[-1] * params.vlad_num_centers])
    output_sq = np.reshape(std, [value.shape[0], value.shape[-1] * params.vlad_num_centers])

    if params.vlad_use_variance:
        output = np.concatenate([output, output_sq], axis=-1)

    if params.vlad_final_l2_norm:
        output /= (np.sqrt(np.sum(output ** 2, axis=1, keepdims=True)) + 1e-16)
    return output, zero, first, second


def compute_phonetic_vlad_wocenter(value, posteriors, params):
    """Compute Phonetic VLAD without centering.
    """
    # The zero-order stat
    zero = np.sum(posteriors, axis=1)
    zero[zero < 1e-16] = 1e-16

    batch_size, length, dim = value.shape
    num_classes = posteriors.shape[2]

    # The first-order stat
    first = np.zeros((batch_size, num_classes, dim))
    for i in range(batch_size):
        for m in range(num_classes):
            for j in range(length):
                first[i, m, :] += (posteriors[i, j, m] * value[i, j, :])
    mean = first / np.expand_dims(zero, axis=2)

    # The second-order stat
    second = np.zeros((batch_size, num_classes, dim))
    for i in range(batch_size):
        for m in range(num_classes):
            for j in range(length):
                second[i, m, :] += (posteriors[i, j, m] * ((value[i, j, :] - mean[i, m, :]) ** 2))
    var = second / np.expand_dims(zero, axis=2)

    # Remove the ghost centers
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var)

    classes = []
    for i in range(num_classes):
        if i not in params.vlad_useless_classes:
            classes.append(i)

    mean = mean[:, classes, :]
    std = std[:, classes, :]
    output = np.reshape(mean, [value.shape[0], mean.shape[1] * mean.shape[2]])
    output_sq = np.reshape(std, [value.shape[0], std.shape[1] * std.shape[2]])

    if params.vlad_use_variance:
        output = np.concatenate([output, output_sq], axis=-1)
    return output, zero, first, second


def compute_lde(value, centers, scale, params):
    """Compute LDE pooling.

    Args:
        value:
        centers:
        scale:
        params:
    :return:
    """
    batch_size, length, dim = value.shape
    ncenter = centers.shape[0]
    res = np.zeros((batch_size, length, ncenter, dim))
    weights = np.zeros((batch_size, length, ncenter))
    for i in range(batch_size):
        for j in range(length):
            for k in range(ncenter):
                res[i, j, k, :] = value[i, j, :] - centers[k, :]
                weights[i, j, k] = np.sum((value[i, j, :] - centers[k, :]) ** 2) * scale[k]

    weights = softmax(weights)
    output = np.zeros((batch_size, length, ncenter, dim))
    for i in range(batch_size):
        for j in range(length):
            for k in range(ncenter):
                output[i, j, k, :] = weights[i, j, k] * res[i, j, k, :]

    output = np.sum(output, axis=1)
    output /= length
    output = np.reshape(output, [batch_size, ncenter * dim])

    if params.lde_use_variance:
        res_sq = np.zeros((batch_size, length, ncenter, dim))
        for i in range(batch_size):
            for j in range(length):
                for k in range(ncenter):
                    res_sq[i, j, k, :] = ((value[i, j, :] - centers[k, :]) ** 2) * weights[i, j, k]
        output_sq = np.sum(res_sq, axis=1) / length
        output_sq[output_sq<1e-12] = 1e-12
        output_sq = np.sqrt(output_sq)
        output_sq = np.reshape(output_sq, [batch_size, ncenter * dim])
        output = np.concatenate([output, output_sq], axis=-1)

    if params.lde_final_l2_norm:
        output /= np.sqrt(np.sum(output ** 2, axis=-1, keepdims=True))
    return output


def pairwise_cos_similarity_np(features):
    """Compute the pairwise cosine similarity.

    Args:
        features: The input embeddings.
    :return: The pairwise cosine similarity matrix.
    """
    feature_norm = np.sqrt(np.sum(features ** 2, axis=1, keepdims=True))
    features = features / feature_norm
    return np.clip(np.dot(features, np.transpose(features)), -1.0, 1.0)


def angular_triplet_proc_positive(sim, margin, loss_type):
    if loss_type =="asoftmax":
        if int(margin) == 1:
            sim = sim
        elif int(margin) == 2:
            if sim > 0:
                k = 0
            else:
                k = 1
            sim = ((-1) ** k) * (np.cos(2 * np.arccos(sim))) - 2 * k
        elif int(margin) == 4:
            l = np.cos(2 * np.arccos(sim))
            if sim > 0 and l > 0:
                k = 0
            elif sim > 0 and l < 0:
                k = 1
            elif sim < 0 and l < 0:
                k = 2
            else:
                k = 3
            sim = ((-1) ** k) * (np.cos(4 * np.arccos(sim))) - 2 * k
        else:
            raise ValueError("Margin in invalid for asoftmax")
    elif loss_type == "amsoftmax":
        sim -= margin
    else:
        if sim <= np.cos(np.pi - margin):
            sim = -np.cos(np.arccos(sim) + margin) - 2
        else:
            sim = np.cos(np.arccos(sim) + margin)
    return sim


def angular_triplet_proc_negative(sim, loss_type):
    return sim


def asoftmax_angular_triplet_loss(features, labels, margin, triplet_type):
    """Compute the triplet loss (using asoftmax loss).

    Args:
        features: The input embeddings.
        labels: The input labels.
        margin: The margin (1, 2, 4).
        triplet_type: all or hard
    :return: The triplet loss
    """
    sim = pairwise_cos_similarity_np(features)
    loss = 0.0
    num_triplets = 0
    num_data = features.shape[0]
    loss_matrix = np.zeros((num_data, num_data, num_data))
    total_loss = np.zeros((num_data, 1))
    eps = 1e-12

    if triplet_type == "all":
        for i in range(num_data):
            for j in range(num_data):
                if i == j or labels[i] != labels[j]:
                    continue
                pos = angular_triplet_proc_positive(sim[i, j], margin, "asoftmax")
                for k in range(num_data):
                    if labels[i] == labels[k]:
                        continue
                    neg = angular_triplet_proc_negative(sim[i, k], "asoftmax")
                    one_loss = neg - pos
                    loss += np.maximum(one_loss, 0.0)
                    loss_matrix[i, j, k] = np.maximum(one_loss, 0.0)
                    if one_loss > eps:
                        num_triplets += 1
    else:
        for i in range(num_data):
            min_pos = 1.0
            for j in range(num_data):
                if labels[i] != labels[j]:
                    continue
                pos = angular_triplet_proc_positive(sim[i, j], margin, "asoftmax")
                if pos < min_pos:
                    min_pos = pos
            max_neg = -1.0
            for j in range(num_data):
                if labels[i] == labels[j]:
                    continue
                neg = angular_triplet_proc_negative(sim[i, j], "asoftmax")
                if neg > max_neg:
                    max_neg = neg
            loss += np.maximum(max_neg - min_pos, 0.0)
            total_loss[i] = np.maximum(max_neg - min_pos, 0.0)
            num_triplets += 1
    return loss / num_triplets


def amsoftmax_angular_triplet_loss(features, labels, margin, triplet_type):
    """Compute the triplet loss (using amsoftmax loss).

    Args:
        features: The input embeddings.
        labels: The input labels.
        margin: The margin.
        triplet_type: all or hard
    :return: The triplet loss
    """
    sim = pairwise_cos_similarity_np(features)
    loss = 0.0
    num_triplets = 0
    num_data = features.shape[0]
    loss_matrix = np.zeros((num_data, num_data, num_data))
    total_loss = np.zeros((num_data, 1))
    eps = 1e-12

    if triplet_type == "all":
        for i in range(num_data):
            for j in range(num_data):
                if i == j or labels[i] != labels[j]:
                    continue
                # pos = angular_triplet_proc_positive(sim[i, j], margin, "amsoftmax")
                for k in range(num_data):
                    if labels[i] == labels[k]:
                        continue
                    # neg = angular_triplet_proc_negative(sim[i, k], "amsoftmax")
                    one_loss = sim[i, k] - sim[i, j] + margin
                    loss += np.maximum(one_loss, 0.0)
                    loss_matrix[i, j, k] = np.maximum(one_loss, 0.0)
                    if one_loss > eps:
                        num_triplets += 1
    else:
        for i in range(num_data):
            min_pos = 1.0
            for j in range(num_data):
                if labels[i] != labels[j]:
                    continue
                pos = angular_triplet_proc_positive(sim[i, j], margin, "amsoftmax")
                if pos < min_pos:
                    min_pos = pos
            max_neg = -1.0
            for j in range(num_data):
                if labels[i] == labels[j]:
                    continue
                neg = angular_triplet_proc_negative(sim[i, j], "amsoftmax")
                if neg > max_neg:
                    max_neg = neg
            loss += np.maximum(max_neg - min_pos, 0.0)
            total_loss[i] = np.maximum(max_neg - min_pos, 0.0)
            num_triplets += 1
    return loss / num_triplets


def arcsoftmax_angular_triplet_loss(features, labels, margin, triplet_type):
    """Compute the triplet loss (using asoftmax loss).

    Args:
        features: The input embeddings.
        labels: The input labels.
        margin: The margin.
        triplet_type: all or hard
    :return: The triplet loss
    """
    sim = pairwise_cos_similarity_np(features)
    loss = 0.0
    num_triplets = 0
    num_data = features.shape[0]
    loss_matrix = np.zeros((num_data, num_data, num_data))
    total_loss = np.zeros((num_data, 1))
    eps = 1e-12

    if triplet_type == "all":
        for i in range(num_data):
            for j in range(num_data):
                if i == j or labels[i] != labels[j]:
                    continue
                pos = angular_triplet_proc_positive(sim[i, j], margin, "arcsoftmax")
                for k in range(num_data):
                    if labels[i] == labels[k]:
                        continue
                    neg = angular_triplet_proc_negative(sim[i, k], "arcsoftmax")
                    one_loss = neg - pos
                    loss += np.maximum(one_loss, 0.0)
                    loss_matrix[i, j, k] = np.maximum(one_loss, 0.0)
                    if one_loss > eps:
                        num_triplets += 1
    else:
        for i in range(num_data):
            min_pos = 1.0
            for j in range(num_data):
                if labels[i] != labels[j]:
                    continue
                pos = angular_triplet_proc_positive(sim[i, j], margin, "arcsoftmax")
                if pos < min_pos:
                    min_pos = pos
            max_neg = -1.0
            for j in range(num_data):
                if labels[i] == labels[j]:
                    continue
                neg = angular_triplet_proc_negative(sim[i, j], "arcsoftmax")
                if neg > max_neg:
                    max_neg = neg
            loss += np.maximum(max_neg - min_pos, 0.0)
            total_loss[i] = np.maximum(max_neg - min_pos, 0.0)
            num_triplets += 1
    return loss / num_triplets


def compute_generalized_triplet_loss(features, w, labels, params, num_classes):
    feature_norm = np.sqrt(np.sum(features ** 2, axis=1, keepdims=True))
    eps = 1e-12

    w_update = w
    if params.triplet_center == "average":
        # Update the centers w
        # w_norm = np.sqrt(np.sum(w ** 2, axis=0, keepdims=True))
        # w_new = w / w_norm
        # for i in range(features.shape[0]):
        #     w_new[:, labels[i]] -= (w_new[:, labels[i]] - np.transpose(features[i, :])) * (
        #             1 - params.triplet_center_momentum)
        # w = w_new
        # w_update = w_new
        w_new = w
        for i in range(features.shape[0]):
            w_new[:, labels[i]] -= (w_new[:, labels[i]] - np.transpose(features[i, :])) * (
                    1 - params.triplet_center_momentum)
            w = w_new
            w_update = w_new

    features = features / feature_norm
    w_norm = np.sqrt(np.sum(w ** 2, axis=0, keepdims=True))
    w = w / w_norm
    cos_theta = np.dot(features, w)
    dist = np.zeros((features.shape[0], w.shape[1]))
    for i in range(features.shape[0]):
        for j in range(w.shape[1]):
            dist[i, j] = np.sum(np.square(features[i, :] - w[:, j]))

    loss = {}
    if params.loss_compute == "raw":
        if params.triplet_topn == 1:
            target_dist = np.zeros((features.shape[0], 1))
            nontarget_dist = np.zeros((features.shape[0], 1))
            for i in range(features.shape[0]):
                min_nontarget_dist = 1e8
                for j in range(num_classes):
                    if j == labels[i]:
                        target_dist[i] = dist[i, j]
                    else:
                        if dist[i, j] < min_nontarget_dist:
                            min_nontarget_dist = dist[i, j]
                nontarget_dist[i] = min_nontarget_dist

            l = 0
            n = 0
            for i in range(target_dist.shape[0]):
                if target_dist[i] > params.target_margin and params.margin + target_dist[i] - nontarget_dist[i] > eps:
                    n += 1
                    l += params.margin + target_dist[i] - nontarget_dist[i]
            loss["triplet_loss"] = l / (n + eps)
        elif params.triplet_topn == 0:
            target_dist = np.zeros((features.shape[0], 1))
            nontarget_dist = np.zeros((features.shape[0], num_classes - 1))
            for i in range(features.shape[0]):
                nontarget_index = 0
                for j in range(num_classes):
                    if j == labels[i]:
                        target_dist[i] = dist[i, j]
                    else:
                        nontarget_dist[i, nontarget_index] = dist[i, j]
                        nontarget_index += 1
            l = 0
            n = 0
            for i in range(features.shape[0]):
                if target_dist[i] > params.target_margin:
                    for j in range(num_classes - 1):
                        if target_dist[i] - nontarget_dist[i, j] + params.margin > eps:
                            l += params.margin + target_dist[i] - nontarget_dist[i, j]
                            n += 1
            loss["triplet_loss"] = l / (n + eps)
        else:
            target_dist = np.zeros((features.shape[0], 1))
            for i in range(features.shape[0]):
                target_dist[i] = dist[i, labels[i]]
            for i in range(features.shape[0]):
                dist[i, labels[i]] = 1e8
            nontarget_dist = np.sort(dist)[:, :params.triplet_topn]

            l = 0
            n = 0
            for i in range(features.shape[0]):
                if target_dist[i] > params.target_margin:
                    for j in range(nontarget_dist.shape[1]):
                        if target_dist[i] - nontarget_dist[i, j] + params.margin > eps:
                            l += params.margin + target_dist[i] - nontarget_dist[i, j]
                            n += 1
            loss["triplet_loss"] = l / (n + eps)

        for i in range(features.shape[0]):
            for j in range(w.shape[1]):
                dist[i, j] = np.sum(np.square(features[i, :] - w[:, j]))
        loss["center_loss"] = 0.0
        n = 0
        for i in range(features.shape[0]):
            if dist[i, labels[i]] > params.target_margin:
                loss["center_loss"] += dist[i, labels[i]]
                n += 1
        loss["center_loss"] /= n

        loss["between_loss"] = 0.0
        num_between = 0
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    continue
                loss["between_loss"] -= np.sum(np.square(w[:, i] - w[:, j]))
                num_between += 1
        loss["between_loss"] /= num_between

        loss["l2_loss"] = 0.0
        for i in range(w_update.shape[0]):
            for j in range(w_update.shape[1]):
                loss["l2_loss"] += w_update[i, j] ** 2
        loss["l2_loss"] = np.sqrt(loss["l2_loss"])
    else:
        # if params.triplet_topn == 1:
        #     # Find the hardest negative
        #     target_cos = np.zeros((features.shape[0], 1))
        #     nontarget_cos = np.zeros((features.shape[0], 1))
        #     for i in range(features.shape[0]):
        #         max_nontarget_cos = -1
        #         for j in range(num_classes):
        #             if j == labels[i]:
        #                 target_cos[i] = cos_theta[i, j]
        #             else:
        #                 if cos_theta[i, j] > max_nontarget_cos:
        #                     max_nontarget_cos = cos_theta[i, j]
        #         nontarget_cos[i] = max_nontarget_cos
        #     l = np.log(1 + np.exp(params.margin + nontarget_cos - target_cos))
        #     if params.triplet_norm_hard:
        #         loss["triplet_loss"] = np.sum(l) / np.sum(np.array(np.greater(params.margin + nontarget_cos - target_cos, eps), dtype=np.float) + eps)
        #     else:
        #         loss["triplet_loss"] = np.sum(l) / features.shape[0]
        # elif params.triplet_topn == 0:
        #     target_cos = np.zeros((features.shape[0], 1))
        #     nontarget_cos = np.zeros((features.shape[0], num_classes - 1))
        #     tmp_loss = np.zeros((features.shape[0], num_classes))
        #     for i in range(features.shape[0]):
        #         for j in range(num_classes):
        #             tmp_loss[i, j] = np.log(1 + np.exp(cos_theta[i, j] - cos_theta[i, labels[i]]))
        #     for i in range(features.shape[0]):
        #         nontarget_index = 0
        #         for j in range(num_classes):
        #             if j == labels[i]:
        #                 target_cos[i] = cos_theta[i, j]
        #             else:
        #                 nontarget_cos[i, nontarget_index] = cos_theta[i, j]
        #                 nontarget_index += 1
        #     loss["triplet_loss"] = 0
        #     num_positive = 0
        #     for i in range(features.shape[0]):
        #         for j in range(num_classes - 1):
        #             if nontarget_cos[i, j] - target_cos[i] + params.margin > eps:
        #                 loss["triplet_loss"] += np.log(1 + np.exp(params.margin + nontarget_cos[i, j] - target_cos[i]))
        #                 num_positive += 1
        #     if params.triplet_norm_hard:
        #         loss["triplet_loss"] /= (num_positive + 1e-16)
        #     else:
        #         loss["triplet_loss"] /= (features.shape[0] * (num_classes - 1))
        #     loss["num_positives"] = num_positive
        # else:
        #     target_cos = np.zeros((features.shape[0], 1))
        #     for i in range(features.shape[0]):
        #         target_cos[i] = cos_theta[i, labels[i]]
        #     for i in range(features.shape[0]):
        #         cos_theta[i, labels[i]] = -1e8
        #     nontarget_cos = np.sort(cos_theta)[:, -params.triplet_topn:]
        #     l = np.log(1 + np.exp(params.margin + nontarget_cos - target_cos))
        #     if params.triplet_norm_hard:
        #         loss["triplet_loss"] = np.sum(l) / np.sum(np.array(np.greater(params.margin + nontarget_cos - target_cos, eps), dtype=np.float) + eps)
        #     else:
        #         loss["triplet_loss"] = np.sum(l) / (features.shape[0] * params.triplet_topn)
        #
        # loss["center_loss"] = 0.0
        # cos_theta = np.dot(features, w)
        # for i in range(features.shape[0]):
        #     loss["center_loss"] += -1 * cos_theta[i, labels[i]]
        # loss["center_loss"] /= features.shape[0]
        #
        # loss["between_loss"] = 0.0
        # num_between = 0
        # for i in range(num_classes):
        #     for j in range(num_classes):
        #         if i == j:
        #             continue
        #         loss["between_loss"] += np.dot(np.transpose(w[:, i]), w[:, j])
        #         num_between += 1
        # loss["between_loss"] /= num_between
        #
        # loss["l2_loss"] = 0.0
        # for i in range(w_update.shape[0]):
        #     for j in range(w_update.shape[1]):
        #         loss["l2_loss"] += w_update[i, j] ** 2
        # loss["l2_loss"] = np.sqrt(loss["l2_loss"])
        pass

    loss["loss"] = loss["triplet_loss"] + params.center_loss_weight * loss["center_loss"] + params.between_loss_weight * loss["between_loss"] + params.l2_loss_weight * loss["l2_loss"]
    return loss, w_update


def compute_ring_loss(features, params, r):
    """ Compute ring loss.

    Args:
        features:
        params:
        r:
    :return:
    """
    return params.ring_loss_lambda * np.sum(np.square(np.abs(np.sqrt(np.sum(features ** 2, axis=1)) - r))) / features.shape[0]


def compute_mhe(labels, params, w):
    """ Compute MHE loss.

    Args:
        labels:
        params:
        w:
    :return:
    """
    loss = 0
    # Norm w
    w /= np.sqrt(np.sum(w ** 2, axis=0, keepdims=True))

    if params.mhe_type == 1:
        cnt = 0
        for i in range(labels.shape[0]):
            for j in range(w.shape[1]):
                if labels[i] == j:
                    continue
                loss += 1.0 / np.dot(np.transpose(w[:, j] - w[:, labels[i]]), (w[:, j] - w[:, labels[i]]))
                cnt += 1
        loss = params.mhe_lambda * (loss / cnt)
    elif params.mhe_type == 2:
        cnt = 0
        for i in range(w.shape[1]):
            for j in range(w.shape[1]):
                if i == j:
                    continue
                loss += 1.0 / np.dot(np.transpose(w[:, j] - w[:, i]), (w[:, j] - w[:, i]))
                cnt += 1
        loss = params.mhe_lambda * (loss / cnt)
    else:
        pass

    return loss


def compute_phone_pooling(features, post):
    # Overall
    p1 = np.concatenate([np.mean(features, axis=1), np.std(features, axis=1)], axis=1)

    batch_size, length, num_classes = post.shape
    z = np.zeros((num_classes, batch_size))
    m = np.zeros((num_classes, batch_size, features.shape[2]))
    s = np.zeros((num_classes, batch_size, features.shape[2]))
    for i in range(num_classes):
        for j in range(batch_size):
            for k in range(length):
                z[i, j] += post[j, k, i]
                m[i, j, :] += post[j, k, i] * features[j, k, :]
            m[i, j, :] /= z[i, j]
    for i in range(num_classes):
        for j in range(batch_size):
            for k in range(length):
                s[i, j, :] += post[j, k, i] * (features[j, k, :] - m[i, j, :]) ** 2
            s[i, j, :] = np.sqrt(s[i, j, :] / z[i, j] + 1e-16)

    results = np.concatenate([p1, np.reshape(np.transpose(np.concatenate([m, s], axis=2), [1,0,2]), [batch_size, -1])], axis=1)
    return results

def compute_sep_phone_pooling(features, post):
    p1 = np.concatenate([np.mean(features, axis=1), np.std(features, axis=1)], axis=1)
    batch_size, length, num_classes = post.shape
    z = np.zeros((num_classes, batch_size))
    m = np.zeros((num_classes, batch_size, features.shape[2]))
    s = np.zeros((num_classes, batch_size, features.shape[2]))
    for i in range(num_classes):
        for j in range(batch_size):
            for k in range(length):
                m[i, j, :] += (post[j, k, i] * features[j, k, :])
                z[i, j] += post[j, k, i]
            m[i, j, :] /= z[i, j]
            for k in range(length):
                s[i, j, :] += (post[j, k, i] * (features[j, k, :] - m[i, j, :]) ** 2)
            s[i, j, :] = np.sqrt(s[i, j, :] / z[i, j] + 1e-16)
    results = np.concatenate([np.expand_dims(p1, 0), np.concatenate([m, s], axis=2)])
    return results

