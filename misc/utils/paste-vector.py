import sys
from dataset.kaldi_io import open_or_fd, read_vec_flt_scp, write_vec_flt
import numpy as np

vector_in = sys.argv[1:-1]
vector_out = sys.argv[-1]

vector_tot = []
vector_names = []

vector_single = {}
for key, vec in read_vec_flt_scp(vector_in[0]):
    vector_names.append(key)
    vector_single[key] = vec
    dim = vec.shape[0]
vector_tot.append(vector_single)

for vector_file in vector_in[1:]:
    vector_single = {}
    index = 0
    for key, vec in read_vec_flt_scp(vector_file):
        assert(key == vector_names[index])
        index += 1
        vector_single[key] = vec
        dim = vec.shape[0]
    vector_tot.append(vector_single)

with open_or_fd(vector_out, 'wb') as f:
    for name in vector_names:
        vector = []
        for vector_single in vector_tot:
            vector.append(vector_single[name])
        vector = np.concatenate(vector, axis=0)
        write_vec_flt(f, vector, key=name)
