import os
import argparse
import shutil
import random
import sys
import tensorflow as tf
import numpy as np
from utils.utils import Params, load_float
from model.trainer import trainer
from dataset.dataset import KaldiDataReader

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cont", action="store_true", help="Continue training from an existing model.")
parser.add_argument("--config", type=str, help="The configuration file.")
parser.add_argument("train_dir", type=str, help="The data directory of the training set.")
parser.add_argument("valid_dir", type=str, help="The data directory of the validation set.")
parser.add_argument("model", type=str, help="The output model directory.")

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()

    # The model directory always has a folder named nnet
    model_dir = os.path.join(args.model, "nnet")
    if args.cont:
        # If we want to continue the model training, we need to check the existence of the checkpoint.
        if not os.path.isdir(os.path.join(args.model, "nnet")):
            sys.exit("To continue training the model, the directory %s must be existed." % (os.path.join(args.model, "nnet")))
        # Simply load the configuration from the saved model.
        params = Params(os.path.join(model_dir, "config.json"))
    else:
        # If we want to train the model from scratch, the model should not exist.
        if os.path.isdir(model_dir):
            sys.exit("The model dir %s exists. Delete it before training" % model_dir)
        params = Params(args.config)
        os.makedirs(model_dir)
        shutil.copyfile(args.config, os.path.join(model_dir, "config.json"))

        # The code may vary in the future, so save the parts which is related to the embedding extraction.
        # TODO: add code to save the code

    # Set the random seed. The random operations may appear in data input, batch forming, etc.
    tf.set_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    # Create dataset before trainer since we need to know the dimension from the dataset.
    train_dataset = KaldiDataReader(args.train_dir, num_parallel=2,
                                    num_speakers=params.num_speakers_per_batch,
                                    num_segments=params.num_segments_per_speaker,
                                    min_len=params.min_segment_len,
                                    max_len=params.max_segment_len)
    # The trainer is used to control the training process
    trainer = trainer(train_dataset.dim, params, args.model)

    if args.cont:
        # If we continue training, we can figure out how much steps the model has been trained,
        # using the index of the checkpoint
        import re
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        else:
            sys.exit("Cannot load checkpoint from %s" % model_dir)
        start_epoch = int(step / params.num_steps_per_epoch)
    else:
        start_epoch = 0

    # In this case, we don't need to create the input nodes in each epoch.
    train_features, train_labels = train_dataset.load()

    # The learning rate is determined by the training process. However, if we continue training, the code doesn't know
    # the previous learning rate if it is tuned using the validation set. To solve that, just save the learning rate to
    # an individual file.
    if os.path.isfile(os.path.join(model_dir, "learning_rate")):
        learning_rate = load_float(os.path.join(model_dir, "learning_rate"))
    else:
        learning_rate = params.learning_rate

    for epoch in range(start_epoch, params.num_epochs):
        trainer.build("train", train_features, train_labels, train_dataset.num_total_speakers)
        trainer.train(learning_rate)

    trainer.close()
