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
