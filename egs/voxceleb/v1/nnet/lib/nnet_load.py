import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help="The model directory.")
parser.add_argument("nnet_wspecifier", type=str, help="nnet output")
args = parser.parse_args()

import tensorflow as tf

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    nnet_dir = os.path.join(args.model_dir, "nnet")

    ckpt = tf.train.get_checkpoint_state(nnet_dir)
    meta_file = ckpt.model_checkpoint_path + ".meta"
    saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
    with tf.Session() as sess:
        variables = tf.trainable_variables()
        saver.restore(sess, ckpt.model_checkpoint_path)
        import pdb
        pdb.set_trace()
        with open(args.nnet_wspecifier, 'w') as f:
            for v in variables:
                vars = sess.run(v)
                f.write("# " + v.name + "\n")
                if len(vars.shape) == 3:
                    for slice in vars:
                        np.savetxt(f, slice)
                        f.write("#")
                else:
                    np.savetxt(f, vars)
                f.write("\n")
