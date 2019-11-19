import argparse
import numpy as np
from dataset.kaldi_io import open_or_fd, read_mat_scp, write_mat

parser = argparse.ArgumentParser()
parser.add_argument("phone_set", type=str, help="The phone set")
parser.add_argument("trans_to_pdf", type=str, help="")
parser.add_argument("phones", type=str, help="")
parser.add_argument("post_in", type=str, help="")
parser.add_argument("post_out", type=str, help="")

if __name__ == '__main__':
    args = parser.parse_args()
    phone_to_pdf = {}
    pdfs = set()
    with open(args.trans_to_pdf, "r") as f:
        f.readline()
        for line in f.readlines():
            transid, pdfid, phoneid, stateid = line.strip().split(" ")
            transid = int(transid)
            pdfid = int(pdfid)
            phoneid = int(phoneid)
            stateid = int(stateid)
            if phoneid not in phone_to_pdf:
                phone_to_pdf[phoneid] = set()
            phone_to_pdf[phoneid].add(pdfid)
            pdfs.add(pdfid)
    pdfs = sorted(list(pdfs))

    phone_id = {}
    with open(args.phones, "r") as f:
        index = 0
        for line in f.readlines():
            p, i = line.strip().split(" ")
            phone_id[p] = index
            assert (index == int(i))
            index += 1

    pdf_to_index = {}
    index_to_pdf = []

    with open(args.phone_set, "r") as f:
        index = 0
        for line in f.readlines():
            phones = line.strip().split(" ")
            tmp = set()
            for p in phones:
                tmp |= phone_to_pdf[phone_id[p]]
                for pdf in phone_to_pdf[phone_id[p]]:
                    if pdf in pdf_to_index:
                        assert (pdf_to_index[pdf] == index)
                    pdf_to_index[pdf] = index
            index_to_pdf.append(list(tmp))
            index += 1
    num_classes = len(index_to_pdf)

    fp_out = open_or_fd(args.post_out, "wb")
    for index, (key, post) in enumerate(read_mat_scp(args.post_in)):
        post_new = np.zeros((post.shape[0], num_classes))
        for i in range(num_classes):
            post_new[:, i] = np.sum(post[:, index_to_pdf[i]], axis=1)
        assert (np.allclose(np.sum(post_new, axis=1), np.ones((post.shape[0], 1)), rtol=1e-02))
        write_mat(fp_out, post_new, key=key)

    fp_out.close()
