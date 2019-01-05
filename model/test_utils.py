import numpy as np


def compute_cos(x1, x2):
    """Compute cosine similarity between x1 and x2"""
    return np.dot(x1, np.transpose(x2)) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-16)


def sigmoid(x):
    """Sigmoid transform."""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
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

