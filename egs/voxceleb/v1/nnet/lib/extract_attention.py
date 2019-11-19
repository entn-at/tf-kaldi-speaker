import argparse
import numpy as np
import os
import sys
import numpy, scipy, sklearn
from model.trainer import Trainer
from misc.utils import Params
from dataset.kaldi_io import FeatureReader, open_or_fd, read_mat_ark, write_mat
from six.moves import range

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=-1, help="The GPU id. GPU disabled if -1.")
parser.add_argument("-m", "--min-chunk-size", type=int, default=25, help="The minimum length of the segments. Any segment shorted than this value will be ignored.")
parser.add_argument("-s", "--chunk-size", type=int, default=10000, help="The length of the segments used to extract the embeddings. "
                                                         "Segments longer than this value will be splited before extraction. "
                                                         "Then the splited embeddings will be averaged to get the final embedding. "
                                                         "L2 normalizaion will be applied before the averaging if specified.")
parser.add_argument("model_dir", type=str, help="The model directory.")
parser.add_argument("rspecifier", type=str, help="Kaldi feature rspecifier (or ark file).")
parser.add_argument("wspecifier", type=str, help="Kaldi output wspecifier (or ark file).")

args = parser.parse_args()

if args.gpu == -1:
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    nnet_dir = os.path.join(args.model_dir, "nnet")
    config_json = os.path.join(args.model_dir, "nnet/config.json")
    if not os.path.isfile(config_json):
        sys.exit("Cannot find params.json in %s" % config_json)
    params = Params(config_json)

    # Attention weights
    params.embedding_node = "attention_weights"

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
            # We only extract the first segment
            feature = feature[:args.chunk_size]
        attention_weights = trainer.predict(feature)
        write_mat(fp_out, attention_weights, key=key)
    fp_out.close()
    trainer.close()
