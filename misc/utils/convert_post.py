import sys
from dataset.kaldi_io import open_or_fd, read_mat_scp, write_mat
import numpy as np

if len(sys.argv) != 4:
    print('Usage: %s phone_class post_in_scp post_out_ark' % sys.argv[0])
    quit()

phone_class = []
with open(sys.argv[1], "r") as f:
    for line in f.readlines():
        phones = line.strip().split(" ")
        phone_class.append([int(p) for p in phones])
num_classes = len(phone_class)

fp_out = open_or_fd(sys.argv[3], "wb")
for key, mat in read_mat_scp(sys.argv[2]):
    post_new = np.zeros((mat.shape[0], num_classes))
    for index, phones in enumerate(phone_class):
        post_new[:, index] = np.sum(mat[:, phones], axis=1)
    write_mat(fp_out, post_new, key=key)

fp_out.close()
