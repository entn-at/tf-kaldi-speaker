import json
import tensorflow as tf
from distutils.dir_util import copy_tree
import os
import sys
import shutil

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class ParamsPlain():
    """Class that saves hyperparameters manually.
    This is used to debug the code since we don't have the json file to feed the parameters.

    Example:
    ```
    params = ParamsPlain()
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self):
        pass

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def save_codes_and_config(args):
    """Save the codes and configuration file.

    During the training, we may modify the codes. It will be problematic when we try to extract embeddings using the old
    model and the new code. So we save the codes when we train the model and use the saved codes to extract embeddings.

    Args:
        args: the arguments parsed from argparse
    :return: A structure params.
    """
    if args.cont:
        # If we want to continue the model training, we need to check the existence of the checkpoint.
        if not os.path.isdir(os.path.join(args.model, "nnet")) or not os.path.isdir(os.path.join(args.model, "codes")):
            sys.exit("To continue training the model, nnet and codes must be existed in %s." % args.model)
        # Simply load the configuration from the saved model.
        tf.logging.info("Continue training from %s." % args.model)
        params = Params(os.path.join(args.model, "nnet/config.json"))
    else:
        # Save the codes in the model directory so that it is more convenient to extract the embeddings.
        # The codes would be changed when we extract the embeddings, making the network loading impossible.
        # When we want to extract the embeddings, we should use the code in `model/codes/...`
        if os.path.isdir(os.path.join(args.model, "nnet")):
            # Backup the codes and configuration in .backup. Keep the model unchanged.
            tf.logging.info("Save backup to %s" % os.path.join(args.model, ".backup"))
            if os.path.isdir(os.path.join(args.model, ".backup")):
                tf.logging.warn("The dir %s exisits. Delete it and continue." % os.path.join(args.model, ".backup"))
                shutil.rmtree(os.path.join(args.model, ".backup"))
            os.makedirs(os.path.join(args.model, ".backup"))
            shutil.move(os.path.join(args.model, "codes"), os.path.join(args.model, ".backup/"))
            shutil.move(os.path.join(args.model, "nnet"), os.path.join(args.model, ".backup/"))

        # `model/codes` is used to save the codes and `model/nnet` is used to save the model and configuration
        os.makedirs(os.path.join(args.model, "codes"))
        copy_tree("../../dataset/", os.path.join(args.model, "codes/dataset/"))
        copy_tree("../../model/", os.path.join(args.model, "codes/model/"))
        copy_tree("../../misc", os.path.join(args.model, "codes/misc/"))
        if not os.path.isdir(os.path.join(args.model, "nnet")):
            os.makedirs(os.path.join(args.model, "nnet"))
        shutil.copyfile(args.config, os.path.join(args.model, "nnet", "config.json"))
        tf.logging.info("Train the model from scratch.")
        params = Params(args.config)
    return params


class ValidLoss():
    """Class that save the valid loss history"""
    def __init__(self):
        self.min_loss = 1e16
        self.min_loss_epoch = -1


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in xrange(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def load_lr(filename):
    """Load learning rate from a saved file"""
    val = 0
    with open(filename, "r") as f:
        for line in f.readlines():
            _, lr = line.strip().split(" ")
            val = float(lr)
    return val


def load_valid_loss(filename):
    """Load valid loss from a saved file"""
    min_loss = ValidLoss()
    with open(filename, "r") as f:
        for line in f.readlines():
            epoch, loss = line.strip().split(" ")
            epoch = int(epoch)
            loss = float(loss)
            if loss < min_loss.min_loss:
                min_loss.min_loss = loss
                min_loss.min_loss_epoch = epoch
    return min_loss


def l2_normalize(x):
    """Normalize the last dimension vector of the input matrix"""
    return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True) + 1e-16)
