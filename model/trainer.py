import tensorflow as tf
import os
import re
import sys
import time
import numpy as np
from model.tdnn import tdnn
from model.loss import softmax


class trainer():
    """Handle the training, validation and prediction"""

    def __init__(self, dim, params, model_dir):
        """
        Args:
            dim: The dimension of the feature
            params: Parameters loaded from JSON.
            model_dir: The model directory.
        """
        self.dim = dim

        # The network configuration
        self.network_type = params.network_type
        if params.network_type == "tdnn":
            self.network = tdnn
        else:
            raise NotImplementedError("Not implement %s network" % params.network_type)
        self.loss_type = params.loss_type
        if params.loss_type == "softmax":
            self.loss_network = softmax
        else:
            raise NotImplementedError("Not implement %s loss" % params.loss_type)

        # We have to save all the parameters since the different models may need different parameters
        self.params = params

        # TODO: add other parameters to the config.
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.sess_config)

        # The model is saved in model/nnet and the evaluation result is saved in model/nnet/eval
        self.model = os.path.join(model_dir, "nnet")

        # The global step. Note that we don't use tf.train.create_global_step because we may extend the code to
        # support adversarial training, in which the global step increases by 1 after `several` updates on the critic
        # and encoder. The internal global_step should be carefully handled in that case. So just a placeholder here,
        # and use a counter to feed in this value is also an option.
        self.global_step = None

        # The learning rate is just a placeholder. I use placeholder because it gives me flexibility to dynamically
        # change the learning rate during training.
        self.learning_rate = None

        # Summary for the training and validation
        self.train_summary = None
        self.valid_summary = None

        # The output predictions. Useful in the prediction mode.
        self.embeddings = None

        # This is only used in validation. Enabling label output give us larger space to do the evaluation
        self.labels = None

        # Training operation. This is called at each step
        self.train_op = None

        # Dicts for training and validation inspection.
        # In the basic condition, the train_ops contains optimization and training loss.
        # And valid loss in the valid_ops. It is possible to add other variables to the dictionaries.
        # Note that the valid loss should be computed from tf.metric.mean, so the valid_ops also has the update ops.
        # In some steps, the train_ops is required to combine with train_summary to get the summary string.
        # These ops are only executed once after several steps (for inspection).
        self.train_ops = {}
        self.valid_ops = {}

        # Model saver and summary writers
        # We don't create the saver or writer here, because after reset, they will be unavailable.
        self.saver = None
        self.summary_writer = None
        self.valid_summary_writer = None

        # This is an indicator to tell whether the model is built. After building the model, we can only use `reuse`
        # to refer to different part of the model.
        self.is_built = False

        # This is only used in prediction. We want to hide details in the class, so we need a placeholder that act as
        # the prediction input. In prediction, we feed the numpy array to this placeholder.
        self.pred_features = None

    def reset(self):
        """Reset the graph so we can create new input pipeline or graph. (Or for other purposes)"""
        try:
            self.sess.close()
        except tf.errors.OpError:
            # Maybe the session is closed before
            pass
        tf.reset_default_graph()
        # The session should be created again after the graph is reset.
        self.sess = tf.Session(config=self.sess_config)
        # After the graph is reset, the flag should be set
        self.is_built = False
        # After reset the graph, it is important to reset the seed.
        tf.set_random_seed(self.params.seed)

    def close(self):
        """Close the session we opened."""
        try:
            self.sess.close()
        except tf.errors.OpError:
            pass

    def load(self):
        """Load the saved variables.

        If the variables have values, the current values will be changed to the saved ones
        :return The step of the saved model.
        """
        tf.logging.info("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            self.saver.restore(self.sess, os.path.join(self.model, ckpt_name))
            tf.logging.info("Succeed to load checkpoint {}".format(ckpt_name))
        else:
            sys.exit("Failed to find a checkpoint in {}".format(self.model))
        return step

    def save(self, step, lr):
        """Save the model.

        Args:
            step: The global step.
            lr: The learning rate. We need to save the learning rate thus next time we can recover the value from the
                checkpoint.
        """
        self.saver.save(self.sess, self.model, global_step=step)
        with open(os.path.join(self.model, "learning_rate"), "w") as f:
            f.write("%f\n" % lr)

    def build(self, mode, features=None, labels=None, num_speakers=None):
        """ Build a network.

        Args:
            mode: `train`, `valid` or `predict`
            features: Tensor or dict of tensors. The input of the network.
            labels: The corresponding labels. `features` and `labels` could be None in the predict mode.
            num_speakers: The total number of speakers. Used in softmax-like network
        """
        assert(mode == "train" or mode == "valid" or mode == "predict")
        is_training = (mode == "train")
        reuse_variables = True if self.is_built else None

        # Create a new path for prediction, since the training may build a tower the support multi-GPUs
        if mode == "predict":
            with tf.name_scope("predict") as scope:
                self.pred_features = tf.placeholder(tf.float32, shape=[None, None, self.dim])
                # There is no need to do L2 normalization in this function, because we can do the normalization outside,
                # or simply a cosine similarity can do it.
                # Note that the output node may be different if we use different loss function. For example, if the
                # softmax is used, the output of 2-last layer is used as the embedding. While if the end2end loss is
                # used, the output of the last layer may be a better choice. So it is impossible to specify the
                # embedding node inside the network structure. The configuration will tell the network to output the
                # correct activations as the embeddings.
                _, endpoints = self.network(self.pred_features, self.params, is_training, reuse_variables)
                self.embeddings = endpoints[self.params.embedding_node]
                if self.saver is None:
                    self.saver = tf.train.Saver()
            return

        if mode == "valid":
            with tf.name_scope("valid") as scope:
                features, endpoints = self.network(features, self.params, is_training, reuse_variables)
                valid_loss = self.loss_network(features, labels, num_speakers, self.params, is_training, reuse_variables)
                # Of course, we can evaluate other stuff in the valid_ops. Just add the new values to the dict.
                # We may also need to check other values expect for the loss. Leave the task to other functions.
                # So I create the embedding output for the validation set thus we can do lots of things with it.
                self.embeddings = endpoints[self.params.embedding_node]
                self.labels = labels

                valid_loss, valid_loss_op = tf.metrics.mean(valid_loss)
                self.valid_ops["valid_loss"] = valid_loss
                self.valid_ops["valid_loss_op"] = valid_loss_op
                self.valid_summary = tf.summary.merge_all(scope=scope)

                if self.saver is None:
                    self.saver = tf.train.Saver(max_to_keep=self.params.keep_checkpoint_max)

                if self.valid_summary_writer is None:
                    self.valid_summary_writer = tf.summary.FileWriter(os.path.join(self.model, "eval"), self.sess.graph)
            return

        self.global_step = tf.placeholder(tf.int32, name="global_step")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # SGD with momentum
        # It is also possible to use other optimizers, e.g. Adam.
        opt = tf.train.MomentumOptimizer(self.learning_rate, self.params.momentum, use_nesterov=self.params.use_nesterov, name="optimizer")

        # Use name_space here. Create multiple name_spaces if multi-gpus
        with tf.name_scope("train") as scope:
            features, endpoints = self.network(features, self.params, is_training, reuse_variables)

            loss = self.loss_network(features, labels, num_speakers, self.params, is_training, reuse_variables)
            regularization_loss = tf.losses.get_regularization_loss()
            total_loss = loss + regularization_loss

            # The training ops is inside the scope to support multi-gpus
            batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
            grads = opt.compute_gradients(total_loss)

            # Once the model has been built (even for a tower), we set the flag
            self.is_built = True
            self.train_summary = tf.summary.merge_all(scope=scope)

        if self.params.clip_gradient:
            grads, vars = zip(*grads)  # compute gradients of variables with respect to loss
            grads_clip, _ = tf.clip_by_global_norm(grads, self.params.clip_gradient_norm)  # l2 norm clipping

            # we follow the instruction in ge2e paper to scale the learning rate for w and b
            if self.loss_type == "ge2e":
                # The parameters w and b must be the last variables in the gradients
                grads_clip = grads_clip[:-2] + [0.01 * grad for grad in grads_clip[-2:]]
                # Simply check the position of w and b
                for var in vars[-2:]:
                    assert("w" in var.name or "b" in var.name)

            grads = zip(grads_clip, vars)

        with tf.control_dependencies(batchnorm_update_ops):
            self.train_op = opt.apply_gradients(grads, global_step=self.global_step)

        # We want to inspect other values during training?
        self.train_ops["loss"] = total_loss
        self.train_ops["raw_loss"] = loss

        # The model saver
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=self.params.keep_checkpoint_max)

        # The training summary writer
        if self.summary_writer is None:
            self.summary_writer = tf.summary.FileWriter(self.model, self.sess.graph)
        return

    def train(self, learning_rate):
        """Train the model.

        Args:
            learning_rate: The learning rate is passed by the main program. The main program can easily tune the
                           learning rate according to the validation accuracy or anything else.
        """
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # curr_step is the real step the training at.
        curr_step = 0

        # Load the model if we have
        if os.path.isfile(os.path.join(self.model, "checkpoint")):
            curr_step = self.load()

        epoch = int(curr_step / self.params.num_steps_per_epoch)
        for step in range(curr_step % self.params.num_steps_per_epoch, self.params.num_steps_per_epoch):
            if step % self.params.save_summary_steps == 0 or step % self.params.show_training_progress == 0:
                train_ops = self.train_ops
                train_ops["train_op"] = self.train_op
                if step % self.params.save_summary_steps == 0:
                    train_ops["train_summary"] = self.train_summary
                start_time = time.time()
                train_val = self.sess.run(train_ops, feed_dict={self.global_step: curr_step,
                                                                self.learning_rate: learning_rate})
                end_time = time.time()
                tf.logging.info(
                    "Epoch: [%2d] step: [%2d/%2d] time: %.4f s/step, raw loss: %f, total loss: %f" \
                    % (epoch, step, self.params.num_steps_per_epoch, end_time - start_time,
                       train_val["raw_loss"], train_val["loss"]))
                if step % self.params.save_summary_steps == 0:
                    self.summary_writer.add_summary(train_val["train_summary"], curr_step)
            else:
                # Only compute optimizer.
                _ = self.sess.run([self.train_op], feed_dict={self.global_step: curr_step,
                                                              self.learning_rate: learning_rate})

            if step % self.params.save_checkpoints_steps == 0 and curr_step != 0:
                self.save(curr_step, learning_rate)
            curr_step += 1
        # Save the model after each epoch
        self.save(curr_step, learning_rate)
        return

    def valid(self, output_embeddings=False):
        """Evaluate on the validation set

        Args:
            output_embeddings: Set True to output the corresponding embeddings and labels of the valid set.
        :return: embeddings and labels (None if output_embeddings is False).
        """
        # Initialization will reset all the variables in the graph
        # The local variables are also need to be initialized for metrics function
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Load the model. The valid function can only be called after training (of course...)
        if os.path.isfile(os.path.join(self.model, "checkpoint")):
            curr_step = self.load()
        else:
            sys.exit("Cannot find model in %s" % self.model)

        valid_ops = self.valid_ops
        valid_ops["valid_summary"] = self.valid_summary
        embeddings_val = None
        labels_val = None
        num_batches = 0
        if output_embeddings:
            # In this mode, the embeddings and labels will be saved and output. It needs more memory and takes longer
            # to process these values.
            valid_ops["embeddings"] = self.embeddings
            valid_ops["labels"] = self.labels
            for _ in xrange(self.params.valid_max_iterations):
                try:
                    if num_batches % 10 == 0:
                        tf.logging.info("valid step: %d" % num_batches)
                    valid_val = self.sess.run(valid_ops)
                    # Save the embeddings and labels
                    if embeddings_val is None:
                        embeddings_val = valid_val["embeddings"]
                        labels_val = valid_val["labels"]
                    else:
                        embeddings_val = np.concatenate((embeddings_val, valid_val["embeddings"]), axis=0)
                        labels_val = np.concatenate((labels_val, valid_val["labels"]), axis=0)
                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break
        else:
            for _ in xrange(self.params.valid_max_iterations):
                try:
                    if num_batches % 10 == 0:
                        tf.logging.info("valid step: %d" % num_batches)
                    valid_val = self.sess.run(valid_ops)
                    num_batches += 1
                except tf.errors.OutOfRangeError:
                    break
        # We only save the summary for the last batch.
        self.valid_summary_writer.add_summary(valid_val['valid_summary'], curr_step)
        # The valid loss is averaged over all the batches.
        tf.logging.info("[Validation %d batches] valid loss: %f" % (num_batches, valid_val["valid_loss"]))

        # The output embeddings and labels can be used to compute EER or other metrics
        return embeddings_val, labels_val

    def predict(self, features):
        """Output the embeddings

        :return: A numpy array which is the embeddings
        """
        if os.path.isfile(os.path.join(self.model, "checkpoint")):
            self.load()
        else:
            sys.exit("Cannot find model in %s" % self.model)
        rank = len(features.shape)
        assert(rank == 2 or rank == 3)
        # Expand the feature if the rank is 2
        if rank == 2:
            features = np.expand_dims(features, axis=0)
        embeddings = self.sess.run(self.embeddings, feed_dict={self.pred_features: features})
        if rank == 2:
            embeddings = np.squeeze(embeddings, axis=0)
        return embeddings
