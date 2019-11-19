import sys
import os
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')

num_systems = len(sys.argv) - 2

colors = ['k', 'b', 'r', 'g', 'm']

def pairwise_dist(weights):
    weights /= np.sqrt(np.sum(weights ** 2, axis=1, keepdims=True))
    dist = 2 - 2 * np.dot(weights, np.transpose(weights))
    return np.reshape(dist, -1)

name = []
dist = []
for i in range(num_systems):
    weights = os.path.join(sys.argv[i+1], 'weights.txt')
    weights = np.loadtxt(weights)
    name.append(sys.argv[i+1].rsplit('/')[-1])
    dist.append(pairwise_dist(weights))

n0_all = []
e0_all = []

plt.figure(1)
for i in range(num_systems):
    n0, e0, _ = plt.hist(dist[i], bins=100, density=True)
    e0 = (e0[:-1] + e0[1:]) / 2
    n0_all.append(n0)
    e0_all.append(e0)
plt.clf()

for i in range(num_systems):
    plt.plot(e0_all[i], n0_all[i], colors[i], label=name[i])
plt.legend()
plt.xlabel("Squared distance", fontsize='x-large')
plt.savefig(sys.argv[-1])
plt.show()
