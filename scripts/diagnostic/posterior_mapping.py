#!/bin/bash
import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=0, help="The index of class to be drawn")
parser.add_argument("post_stat", type=str, help="")
parser.add_argument("plot_dir", type=str, help="")


args = parser.parse_args()

index_stat = []
num_acc = 0
num_frames = 0
num_classes = 0

with open(args.post_stat, "r") as f:
    a, n = f.readline().strip().split(" ")
    num_acc = int(a)
    num_frames = int(n)
    c = f.readline().strip()
    num_classes = int(c)
    assert(args.n < num_classes)

    for index, line in enumerate(f.readlines()):
        if len(index_stat) <= index:
            # Add a new entry
            index_stat.append([0, 0])

        tmp = line.strip().split(" ")
        count = int(tmp[0])
        if count == 0:
            continue
        index_stat[index][0] += count
        post = np.array([float(t) for t in tmp[1:]])
        index_stat[index][1] += post

if not os.path.isdir(args.plot_dir):
    os.makedirs(args.plot_dir)

for i in range(num_classes):
    if index_stat[i][0] == 0:
        print("No occureence for class %d" % i)
        continue
    dim = index_stat[i][1].shape[0]
    assert dim == num_classes
    if np.argmax(index_stat[i][1]) != i:
        print("Class %d has maximum at class %d" % (i, np.argmax(index_stat[i][1])))

plt.figure(1)
norm = index_stat[args.n][1] / index_stat[args.n][0]
# There is a bug in plt.bar that too many bars will make some bars disappear.
plt.bar(range(num_classes), norm, linewidth=5)
plt.xlim([-num_classes/10, num_classes])
plt.ylim([0, np.max(norm)*1.1])
plt.savefig(os.path.join(args.plot_dir, str(args.n)))
plt.show()
