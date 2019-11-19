import sys
import numpy as np
from dataset.kaldi_io import read_vec_flt_scp

ivec_scp = sys.argv[1]
plda_ood = sys.argv[2]

output_mat = sys.argv[3]


ivectors = []
for key, vec in read_vec_flt_scp(ivec_scp):
    ivectors.append(vec)
ivectors = np.array(ivectors)
np.savetxt(output_mat, ivectors)
