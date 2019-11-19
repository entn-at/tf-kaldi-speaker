# Compute the accuracy of the posteriors. Compute the correct and total numbers of frames.
# Sum up senones posteriors if needed.
# Compute the posterior summation for all phonetic classes and compute the occurrence for the classes
# Write the statistics in a txt file.
import numpy as np
import argparse
from dataset.kaldi_io import open_or_fd, read_mat_scp, read_key, read_vec_int

parser = argparse.ArgumentParser()
parser.add_argument("--phone-set", type=str, default=None, help="The phonetic set to cluster phones.")
parser.add_argument("phones", type=str, help="phones.txt")
parser.add_argument("trans_to_pdf", type=str, help="transition_id to pdf_id and phone_id")
parser.add_argument("ali", type=str, help="The alignment ark")
parser.add_argument("post", type=str, help="The posteriors ark")
parser.add_argument("stat", type=str, help="The output statitics")

def read_ali(fd):
    key = read_key(fd)
    ali = None
    if key:
        ali = read_vec_int(fd)
    return key, ali


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
            assert(index == int(i))
            index += 1

    pdf_to_index = {}
    index_to_pdf = []
    if args.phone_set is None:
        for index, pdf in enumerate(pdfs):
            assert(index == pdf)
            pdf_to_index[pdf] = index
            index_to_pdf.append([pdf])
    else:
        with open(args.phone_set, "r") as f:
            index = 0
            for line in f.readlines():
                phones = line.strip().split(" ")
                tmp = set()
                for p in phones:
                    tmp |= phone_to_pdf[phone_id[p]]
                    for pdf in phone_to_pdf[phone_id[p]]:
                        if pdf in pdf_to_index:
                            assert(pdf_to_index[pdf] == index)
                        pdf_to_index[pdf] = index
                index_to_pdf.append(list(tmp))
                index += 1

    print("%d classes" % len(index_to_pdf))
    num_classes = len(index_to_pdf)

    # Sanity check
    for pdf in pdf_to_index:
        assert(pdf in index_to_pdf[pdf_to_index[pdf]])

    index_stat = {}  # index -> [count, post]
    num_acc = 0
    num_frames = 0

    dim = 0
    num_err = 0
    num_done = 0
    fp_ali = open_or_fd(args.ali)
    ali_key, ali_value = read_ali(fp_ali)
    for index, (key, post) in enumerate(read_mat_scp(args.post)):
        if ali_key != key:
            num_err += 1
            continue

        # Main computation
        # Convert alignment
        ali_value_new = [pdf_to_index[a] for a in ali_value]

        # Convert posteriors
        post_new = np.zeros((post.shape[0], num_classes))
        for i in range(num_classes):
            post_new[:, i] = np.sum(post[:, index_to_pdf[i]], axis=1)
        assert(np.allclose(np.sum(post_new, axis=1), np.ones((post.shape[0], 1)), rtol=1e-02))

        num_acc += np.sum(np.argmax(post_new, axis=1) == np.array(ali_value_new))
        num_frames += post.shape[0]

        for i in range(post.shape[0]):
            if ali_value_new[i] not in index_stat:
                dim = post_new.shape[1]
                index_stat[ali_value_new[i]] = [0, np.zeros((1, dim))]
            index_stat[ali_value_new[i]][0] += 1
            index_stat[ali_value_new[i]][1] += post_new[i, :]

        num_done += 1
        ali_key, ali_value = read_ali(fp_ali)
        if ali_key is None:
            break

    fp_ali.close()

    if float(num_done) / (num_done + num_err) < 0.8:
        raise IOError("Many utterances are failed. Please check the files.")
    print("Processed %d utterances, %d has errors" % (num_done + num_err, num_err))
    print("Frame accuracy: %f" % (float(num_acc) / num_frames))

    # Write the statistics
    # Format:
    # num_acc num_frames
    # num_classes
    # occ1 post1
    # occ2 post2
    # ...
    with open(args.stat, "w") as f:
        f.write("%d %d\n" % (num_acc, num_frames))
        f.write("%d\n" % num_classes)
        for i in range(num_classes):
            if i not in index_stat:
                f.write("0\n")
                continue
            f.write("%d" % index_stat[i][0])
            for j in range(dim):
                f.write(" %e" % index_stat[i][1][0, j])
            f.write("\n")





