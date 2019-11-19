import argparse
import numpy as np
import os
import sys
import numpy, scipy, sklearn
from model.trainer import Trainer
from misc.utils import Params
from dataset.kaldi_io import FeatureReader, open_or_fd, read_mat_scp, write_mat
from six.moves import range
from dataset.data_loader import KaldiDataRandomQueue, KaldiDataSeqQueue, DataOutOfRange


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=-1, help="The GPU id. GPU disabled if -1.")
parser.add_argument("-m", "--min-chunk-size", type=int, default=25, help="The minimum length of the segments. Any segment shorted than this value will be ignored.")
parser.add_argument("-s", "--chunk-size", type=int, default=10000, help="The length of the segments used to extract the embeddings. "
                                                         "Segments longer than this value will be splited before extraction. "
                                                         "Then the splited embeddings will be averaged to get the final embedding. "
                                                         "L2 normalizaion will be applied before the averaging if specified.")
parser.add_argument("--init", action="store_true", help="load the initial model")
parser.add_argument("model_dir", type=str, help="The model directory.")
parser.add_argument("rspecifier", type=str, help="Kaldi feature scp file.")
parser.add_argument("utt2spk", type=str, help="utt2spk")
parser.add_argument("spklist", type=str, help="spklist")
parser.add_argument("angles", type=str, help="output angles")

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

    # First, we need to extract the weights
    num_total_train_speakers = KaldiDataRandomQueue(os.path.dirname(args.spklist), args.spklist).num_total_speakers
    dim = FeatureReader(os.path.dirname(args.spklist)).get_dim()
    if "selected_dim" in params.dict:
        dim = params.selected_dim
    trainer = Trainer(params, args.model_dir, dim, num_total_train_speakers, single_cpu=True)
    trainer.build("valid")
    trainer.sess.run(tf.global_variables_initializer())
    trainer.sess.run(tf.local_variables_initializer())

    if not args.init:
        curr_step = trainer.load()
    else:
        # Hack:
        tf.logging.info("Use random initialization")
        trainer.is_loaded = True

    with tf.variable_scope("softmax", reuse=True):
        kernel = tf.get_variable("output/kernel", shape=[trainer.embeddings.get_shape()[-1], num_total_train_speakers])
        kernel_val = trainer.sess.run(kernel)
    weights = np.transpose(kernel_val)

    # Output the final activation (prior to the softmax layer)
    params.embedding_node = "output"
    trainer.build("predict")

    if args.rspecifier.rsplit(".", 1)[1] != "scp":
        # The rspecifier cannot be scp
        sys.exit("The rspecifier must be scp")

    spk2int = {}
    with open(args.spklist, 'r') as f:
        for line in f.readlines():
            spk, i = line.strip().split(" ")
            spk2int[spk] = int(i)

    utt2spk = {}
    with open(args.utt2spk, 'r') as f:
        for line in f.readlines():
            utt, spk = line.strip().split(" ")
            utt2spk[utt] = spk

    fp_out = open(args.angles, 'w')
    for key, feature in read_mat_scp(args.rspecifier):
        i = spk2int[utt2spk[key]]
        if feature.shape[0] < args.min_chunk_size:
            tf.logging.info("[INFO] Key %s length too short, %d < %d, skip." % (key, feature.shape[0], args.min_chunk_size))
            continue
        if feature.shape[0] > args.chunk_size:
            feature = feature[:args.chunk_size]
        output = trainer.predict(feature)
        angle = np.dot(weights[i, :], np.transpose(output)) / (np.sqrt(np.sum(weights[i, :] ** 2)) * np.sqrt(np.sum(output ** 2)))
        angle = np.arccos(angle)
        fp_out.write("%s %f\n" % (key, angle))
    fp_out.close()
    trainer.close()





