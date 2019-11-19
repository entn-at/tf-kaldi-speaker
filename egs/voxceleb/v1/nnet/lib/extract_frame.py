import argparse
import numpy as np
import os
import sys
import numpy, scipy, sklearn
from model.trainer import Trainer
from misc.utils import Params
from dataset.kaldi_io import open_or_fd, read_mat_ark, write_mat
from six.moves import range

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=-1, help="The GPU id. GPU disabled if -1.")
parser.add_argument("-m", "--min-chunk-size", type=int, default=25, help="The minimum length of the segments. Any segment shorted than this value will be ignored.")
parser.add_argument("-s", "--chunk-size", type=int, default=10000, help="The length of the segments used to extract the embeddings. "
                                                         "Segments longer than this value will be splited before extraction. "
                                                         "Then the splited embeddings will be averaged to get the final embedding. "
                                                         "L2 normalizaion will be applied before the averaging if specified.")
parser.add_argument("--node", type=str, default="", help="The node to output the embeddings.")
parser.add_argument("model_dir", type=str, help="The model directory.")
parser.add_argument("rspecifier", type=str, help="Kaldi feature rspecifier (or ark file).")
parser.add_argument("wspecifier", type=str, help="Kaldi output wspecifier (or ark file).")

args = parser.parse_args()

if args.gpu == -1:
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# In the GPU situation, it is difficult to know how to specify the GPU id.
# If the program is launched locally, you can set CUDA_VISIBLE_DEVICES to the id.
# However, if SGE is used, we cannot simply set CUDA_VISIBLE_DEVICES.
# So it is better to specify the GPU id outside the program.
# Give an arbitrary number (except for -1) to --gpu can enable it. Leave it blank if you want to disable gpu.

import tensorflow as tf

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    nnet_dir = os.path.join(args.model_dir, "nnet")

    config_json = os.path.join(args.model_dir, "nnet/config.json")
    if not os.path.isfile(config_json):
        sys.exit("Cannot find params.json in %s" % config_json)
    params = Params(config_json)

    # Change the output node if necessary
    if len(args.node) != 0:
        params.embedding_node = args.node
    tf.logging.info("Extract embedding from %s" % params.embedding_node)

    with open(os.path.join(nnet_dir, "feature_dim"), "r") as f:
        dim = int(f.readline().strip())
    trainer = Trainer(params, args.model_dir, dim, single_cpu=True)
    trainer.build("predict")

    if args.rspecifier.rsplit(".", 1)[1] == "scp":
        # The rspecifier cannot be scp
        sys.exit("The rspecifier must be ark or input pipe")

    fp_out = open_or_fd(args.wspecifier, "wb")
    for index, (key, feature) in enumerate(read_mat_ark(args.rspecifier)):
        if feature.shape[0] < args.min_chunk_size:
            tf.logging.info("[INFO] Key %s length too short, %d < %d, skip." % (key, feature.shape[0], args.min_chunk_size))
            continue
        if feature.shape[0] > args.chunk_size:
            feature_array = []
            feature_length = []
            num_chunks = int(np.ceil(float(feature.shape[0]) / args.chunk_size))
            tf.logging.info("[INFO] Key %s length %d > %d, split to %d segments." % (key, feature.shape[0], args.chunk_size, num_chunks))
            for i in range(num_chunks):
                start = i * args.chunk_size
                this_chunk_size = args.chunk_size if feature.shape[0] - start > args.chunk_size else feature.shape[0] - start
                feature_length.append(this_chunk_size)
                feature_array.append(feature[start:start+this_chunk_size])

            # Except for the last feature, the length of other features should be the same (=chunk_size)
            # TODO: has not been checked
            embeddings = trainer.predict(np.array(feature_array[:-1], dtype=np.float32))
            pad = (feature_array[0].shape[0] - embeddings.shape[1]) / 2
            embeddings = np.concatenate([np.tile(embeddings[:, 0, :], [1, pad, 1]), embeddings, np.tile(embeddings[:, -1, :], [1, pad, 1])], axis=1)
            embedding_last = trainer.predict(feature_array[-1])
            pad = (feature_array[-1].shape[0] - embedding_last.shape[0]) / 2
            embedding_last = np.concatenate([np.tile(embedding_last[0, :], [pad, 1]), embedding_last, np.tile(embedding_last[-1, :], [pad, 1])], axis=0)
            embeddings = np.reshape(embeddings, [embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2]])
            embedding = np.concatenate([embeddings, embedding_last], axis=0)
        else:
            tf.logging.info("[INFO] Key %s length %d." % (key, feature.shape[0]))
            embedding = trainer.predict(feature)
            # padding
            pad = (feature.shape[0] - embedding.shape[0]) / 2
            embedding = np.concatenate([np.tile(embedding[0, :], [pad, 1]), embedding, np.tile(embedding[-1, :], [pad, 1])], axis=0)

        assert(embedding.shape[0] == feature.shape[0])
        write_mat(fp_out, embedding, key=key)
    fp_out.close()
    trainer.close()
