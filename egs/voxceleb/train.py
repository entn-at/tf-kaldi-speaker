import os
import argparse
import random
import sys
import tensorflow as tf
import numpy as np
from misc.utils import ValidLoss, load_lr, load_valid_loss, save_codes_and_config
from model.trainer import Trainer
from dataset.data_loader import KaldiDataRandomQueue
from dataset.light_kaldi_io import FeatureReader

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cont", action="store_true", help="Continue training from an existing model.")
parser.add_argument("--config", type=str, help="The configuration file.")
parser.add_argument("train_dir", type=str, help="The data directory of the training set.")
parser.add_argument("train_spklist", type=str, help="The spklist file maps the TRAINING speakers to the indices.")
parser.add_argument("valid_dir", type=str, help="The data directory of the validation set.")
parser.add_argument("valid_spklist", type=str, help="The spklist maps the VALID speakers to the indices.")
parser.add_argument("model", type=str, help="The output model directory.")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()
    params = save_codes_and_config(args)

    # The model directory always has a folder named nnet
    model_dir = os.path.join(args.model, "nnet")

    # Set the random seed. The random operations may appear in data input, batch forming, etc.
    tf.set_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

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

    # The learning rate is determined by the training process. However, if we continue training, the code doesn't know
    # the previous learning rate if it is tuned using the validation set. To solve that, just save the learning rate to
    # an individual file.
    if os.path.isfile(os.path.join(model_dir, "learning_rate")):
        learning_rate = load_lr(os.path.join(model_dir, "learning_rate"))
    else:
        learning_rate = params.learning_rate

    dim = FeatureReader(args.train_dir).get_dim()
    num_total_train_speakers = KaldiDataRandomQueue(args.train_dir, args.train_spklist).num_total_speakers
    tf.logging.info("There are %d speakers in the training set and the dim is %d" % (num_total_train_speakers, dim))

    # Load the history valid loss
    min_valid_loss = ValidLoss()
    if os.path.isfile(os.path.join(model_dir, "valid_loss")):
        min_valid_loss = load_valid_loss(os.path.join(model_dir, "valid_loss"))

    # The trainer is used to control the training process
    trainer = Trainer(params, args.model)
    trainer.build("train",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)
    trainer.build("valid",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)
    # You can tune the learning rate using the following function.
    # After training, you should plot the loss v.s. the learning rate and pich a learning rate that decrease the
    # loss fastest.
    trainer.train_tune_lr(args.train_dir, args.train_spklist)
    sys.exit("Finish tuning.")

    for epoch in range(start_epoch, params.num_epochs):
        trainer.train(args.train_dir, args.train_spklist, learning_rate)
        valid_loss, _, _ = trainer.valid(args.valid_dir, args.valid_spklist, batch_type=params.batch_type)

        # Tune the learning rate
        if valid_loss < min_valid_loss.min_loss:
            min_valid_loss.min_loss = valid_loss
            min_valid_loss.min_loss_epoch = epoch
        else:
            if epoch - min_valid_loss.min_loss_epoch >= params.reduce_lr_epochs:
                learning_rate /= 2
                # Wait for an extra epochs to see the loss reduction.
                min_valid_loss.min_loss_epoch = epoch - params.reduce_lr_epochs + 1
                tf.logging.info("After epoch %d, no improvement. Reduce the learning rate to %f" % (min_valid_loss.min_loss_epoch, learning_rate))

        # Save the learning rate and loss for each epoch.
        with open(os.path.join(model_dir, "learning_rate"), "a") as f:
            f.write("%d %f\n" % (epoch, learning_rate))
        with open(os.path.join(model_dir, "valid_loss"), "a") as f:
            f.write("%d %f\n" % (epoch, valid_loss))

    # Close the session before we exit.
    trainer.close()
