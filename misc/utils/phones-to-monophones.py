import sys
from dataset.kaldi_io import open_or_fd, read_vec_int_scp, write_vec_int
import numpy as np

if len(sys.argv) != 4:
    print("usage: %s phones.txt phone_scp monophone_ark" % sys.argv[0])
    quit()

phones = sys.argv[1]
phone_scp = sys.argv[2]
monophone_ark = sys.argv[3]

ind = 0
phone2ind = {}
num2ind = {}

with open(phones, 'r') as f:
    for line in f.readlines():
        phone, num = line.strip().split(" ")
        num = int(num)
        p = phone.rsplit("_", 1)[0]
        if p not in phone2ind:
            phone2ind[p] = ind
            ind += 1
        num2ind[num] = phone2ind[p]

ind2cnt = {}
fp_out = open_or_fd(monophone_ark, "wb")
for key, vec in read_vec_int_scp(phone_scp):
    a = []
    for v in vec:
        a.append(num2ind[v])
        if num2ind[v] not in ind2cnt:
            ind2cnt[num2ind[v]] = 0
        ind2cnt[num2ind[v]] += 1
    write_vec_int(fp_out, np.array(a), key=key)

fp_out.close()
for ind in ind2cnt:
    print("%d %d" % (ind, ind2cnt[ind]))