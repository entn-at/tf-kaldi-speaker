import argparse
from dataset.kaldi_io import read_vec_flt, read_vec_int, read_mat
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=10, help="")
parser.add_argument("--alpha", type=float, default=0, help="The weight between the alignment and the posterior")
parser.add_argument("--stat", type=str, default=None, help="posterior statistics")
parser.add_argument("--phone-set", type=str, default=None, help="monophone.txt")
parser.add_argument("trans_to_pdf", type=str, help="trans-id_to_pdf-id")
parser.add_argument("phones", type=str, help="phones.txt")
parser.add_argument("ali_scp", type=str, help="ali_pdf.scp")
parser.add_argument("post_scp", type=str, help="post.scp")
parser.add_argument("xvector_scp", type=str, help="xvector.scp")


def get_pdf_info(phone_set, trans_to_pdf, phones):
    phone_to_pdf = {}
    pdfs = set()
    with open(trans_to_pdf, "r") as f:
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
    with open(phones, "r") as f:
        index = 0
        for line in f.readlines():
            p, i = line.strip().split(" ")
            phone_id[p] = index
            assert (index == int(i))
            index += 1

    pdf_to_index = {}
    index_to_pdf = []
    if phone_set is None:
        for index, pdf in enumerate(pdfs):
            assert (index == pdf)
            pdf_to_index[pdf] = index
            index_to_pdf.append([pdf])
    else:
        with open(phone_set, "r") as f:
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

    print("%d classes" % len(index_to_pdf))
    return index_to_pdf, pdf_to_index


def main(args):
    n_embeddings = args.n
    alpha = args.alpha
    map_stat = args.stat

    index_to_pdf, pdf_to_index = get_pdf_info(args.phone_set, args.trans_to_pdf, args.phones)
    num_classes = len(index_to_pdf)

    utts = []
    embeddings_reader = {}
    ali_reader = {}
    post_reader = {}
    num_correct = 0
    num_total = 0

    with open(args.xvector_scp, "r") as f:
        for line in f.readlines():
            utt, l = line.strip().split(" ")
            embeddings_reader[utt] = l
            utts.append(utt)

    with open(args.ali_scp, "r") as f:
        for line in f.readlines():
            utt, l = line.strip().split(" ")
            ali_reader[utt] = l

    with open(args.post_scp, "r") as f:
        for line in f.readlines():
            utt, l = line.strip().split(" ")
            post_reader[utt] = l

    index_post = []
    if map_stat is not None:
        with open(map_stat, "r") as f:
            f.readline().strip().split(" ")
            c = f.readline().strip()
            assert (int(c) == num_classes)
            for index, line in enumerate(f.readlines()):
                tmp = line.strip().split(" ")
                count = int(tmp[0])
                if count == 0:
                    raise ValueError("Class %d has no occurrence." % index)
                p = np.array([float(t) for t in tmp[1:]])
                index_post.append(p / count)

    for utt in utts[:n_embeddings]:
        if utt not in post_reader or utt not in ali_reader:
            continue
        embedding = read_mat(embeddings_reader[utt])
        ali = read_vec_int(ali_reader[utt])
        post = read_mat(post_reader[utt])

        ali_new = [pdf_to_index[a] for a in ali]
        post_new = np.zeros((post.shape[0], num_classes))
        for i in range(num_classes):
            post_new[:, i] = np.sum(post[:, index_to_pdf[i]], axis=1)

        # # randomly generate the alignments and posteriors
        # ali_new = np.random.randint(num_classes, size=(len(ali_new)))
        # post_new = np.array(np.random.randint(100, size=(post_new.shape[0], num_classes)), dtype=np.float)
        # post_new = post_new / np.sum(post_new, axis=1, keepdims=True)

        # overall mean and std
        mean = np.mean(embedding, axis=0)
        var = np.var(embedding, axis=0)
        metric = np.sum(var)

        # for each class
        mean_class = np.zeros((num_classes, embedding.shape[1]))
        var_class = np.zeros((num_classes, embedding.shape[1]))
        count = np.zeros((num_classes))
        for i in range(post_new.shape[0]):
            if map_stat is None:
                p = np.zeros((num_classes))
                p[ali_new[i]] = 1
                p = alpha * p + (1-alpha) * post_new[i, :]
            else:
                # posterior mapping
                p = index_post[ali_new[i]]

            assert (abs(np.sum(p) - 1) < 1e-4)
            count += p
            mean_class += p[:, np.newaxis] * embedding[i, :][np.newaxis, :]
            var_class += p[:, np.newaxis] * (embedding[i, :][np.newaxis, :] ** 2)
            if np.argmax(p) == ali_new[i]:
                num_correct += 1
            num_total += 1

        mean_class /= (count[:, np.newaxis] + 1e-16)
        var_class /= (count[:, np.newaxis] + 1e-16)
        var_class -= (mean_class ** 2)
        metric_class = np.sum(var_class, axis=1)

        print("%s" % utt)
        print(metric)
        print(metric_class)
    print("Acc: %.4f" % (float(num_correct) / num_total))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
