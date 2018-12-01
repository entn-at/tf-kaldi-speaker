import numpy as np


def compute_cos(x1, x2):
    """Compute cosine similarity between x1 and x2"""
    return np.dot(x1, np.transpose(x2)) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-16)


def sigmoid(x):
    """Sigmoid transform."""
    return 1 / (1 + np.exp(-x))


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
    embeddings /= np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))
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
                sim[i, j] = w * compute_cos(embeddings[i, :], center_exclusive / (n_exclusive_samples + 1e-16)) + b
            else:
                sim[i, j] = w * compute_cos(embeddings[i, :], centers[j, :]) + b

    n_samples, n_classes = sim.shape
    loss = 0

    # Find single points
    class2count = {}
    for i in class_index:
        if i not in class2count:
            class2count[i] = 0
        class2count[i] += 1

    cnt = 0
    if ge2e_type == "softmax":
        for i in range(n_samples):
            loss -= sim[i, class_index[i]] - np.log(np.sum(np.exp(sim[i, :])) + 1e-16)
            cnt += 1
    else:
        for i in range(n_samples):
            other = [0]
            for j in range(n_classes):
                if class_index[i] == j:
                    continue
                other.append(sigmoid(sim[i, j]))
            other = sorted(other)
            loss += 1 - sigmoid(sim[i, class_index[i]]) + other[-1]
            cnt += 1
    return loss


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
