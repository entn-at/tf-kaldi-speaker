import os
import argparse
import random
import sys
import tensorflow as tf
import numpy as np
from misc.utils import ValidLoss, load_lr, load_valid_loss, save_codes_and_config, get_pretrain_model
from model.trainer import Trainer
from dataset.data_loader import KaldiDataRandomQueue
from dataset.kaldi_io import FeatureReader

# We don't need to use a `continue` option here, because if we want to resume training, we should simply use train.py,
# since we want the model to restore everything rather than a part of the graph, which is the case when we begin to
# fine-tune the model.
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=int, default=-1, help="The checkpoint as the pre-trained model (default: the last one)")
parser.add_argument("--config", type=str, help="The configuration file.")
parser.add_argument("train_dir", type=str, help="The data directory of the training set.")
parser.add_argument("train_spklist", type=str, help="The spklist file maps the TRAINING speakers to the indices.")
parser.add_argument("valid_dir", type=str, help="The data directory of the validation set.")
parser.add_argument("valid_spklist", type=str, help="The spklist maps the VALID speakers to the indices.")
parser.add_argument("pretrain_model", type=str, help="The pre-trained model directory.")
parser.add_argument("finetune_model", type=str, help="The fine-tuned model directory")


def compute_pairwise_eer(embeddings, labels):
    """Compute EER. The target and nontarget scores are 25 to 75 percentile of all the pairwise scores
    """
    from sklearn import metrics
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    import numpy as np
    embeddings /= np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))

    max_num_embeddings = 5000
    num_embeddings = embeddings.shape[0]
    if num_embeddings > max_num_embeddings:
        # Downsample the embeddings and labels
        step = num_embeddings / max_num_embeddings
        embeddings = embeddings[range(0, num_embeddings, step), :]
        labels = labels[range(0, num_embeddings, step)]

    spk2embeddings = {}
    for i, l in enumerate(labels):
        if l not in spk2embeddings:
            spk2embeddings[l] = []
        spk2embeddings[l].append(embeddings[i])
    for spk in spk2embeddings:
        spk2embeddings[spk] = np.array(spk2embeddings[spk])

    target_scores = []
    nontarget_scores = []
    spks = list(spk2embeddings.keys())
    for spk in spks:
        scores = np.dot(spk2embeddings[spk], np.transpose(spk2embeddings[spk]))
        num_segs = spk2embeddings[spk].shape[0]
        sscores = []
        for i in range(num_segs - 1):
            for j in range(i+1, num_segs):
                sscores.append(scores[i, j])
        sscores = sorted(sscores)
        n_scores = len(sscores)
        target_scores += sscores[(n_scores / 4):(3 * n_scores / 4)]

    for i in range(len(spks)):
        for j in range(i):
            scores = np.dot(np.array(spk2embeddings[spks[i]]), np.transpose(np.array(spk2embeddings[spks[j]])))
            scores = np.reshape(scores, (-1,))
            scores = np.sort(scores)
            n_scores = scores.shape[0]
            nontarget_scores += scores[(n_scores / 4):(3 * n_scores / 4)].tolist()

    keys = len(target_scores) * [1] + len(nontarget_scores) * [0]
    scores = target_scores + nontarget_scores
    fpr, tpr, thresholds = metrics.roc_curve(keys, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()

    first_stage_model = os.path.join(args.finetune_model, "stage_one")
    second_stage_model = os.path.join(args.finetune_model, "stage_two")

    dim = FeatureReader(args.train_dir).get_dim()
    num_total_train_speakers = KaldiDataRandomQueue(args.train_dir, args.train_spklist).num_total_speakers
    tf.logging.info("There are %d speakers in the training set and the dim is %d" % (num_total_train_speakers, dim))

    # Stage 1:
    # # Load the pretrained model to the target model directory.
    # params = save_codes_and_config(False, first_stage_model, args.config)
    #
    # # Set the random seed. The random operations may appear in data input, batch forming, etc.
    # tf.set_random_seed(params.seed)
    # random.seed(params.seed)
    # np.random.seed(params.seed)
    # # The model directory always has a folder named nnet
    # get_pretrain_model(args.checkpoint,
    #                    os.path.join(args.pretrain_model, "nnet"),
    #                    os.path.join(first_stage_model, "nnet"))
    # model_dir = os.path.join(first_stage_model, "nnet")
    # with open(os.path.join(model_dir, "feature_dim"), "w") as f:
    #     f.write("%d\n" % dim)
    # # The trainer is used to control the training process
    # trainer = Trainer(params, first_stage_model)
    # trainer.build("train",
    #               dim=dim,
    #               loss_type=params.loss_func,
    #               num_speakers=num_total_train_speakers)
    # trainer.build("valid",
    #               dim=dim,
    #               loss_type=params.loss_func,
    #               num_speakers=num_total_train_speakers)
    #
    # # Load the pre-trained model and transfer to current model
    # trainer.get_finetune_model(params.exclude_params_list)
    #
    # # First, we train the last layer which is randomly initialized.
    # trainer.set_trainable_variables(params.exclude_params_list)
    # for epoch in range(0, params.num_first_epochs):
    #     # Before any further training, we evaluate the performance of the current model
    #     valid_loss, valid_embeddings, valid_labels = trainer.valid(args.valid_dir, args.valid_spklist,
    #                                                                batch_type=params.batch_type,
    #                                                                output_embeddings=True)
    #     eer = compute_pairwise_eer(valid_embeddings, valid_labels)
    #     tf.logging.info("[INFO] Valid EER: %f" % eer)
    #
    #     trainer.train(args.train_dir, args.train_spklist, params.learning_rate)
    #     # The learning rate may be tuned manually.
    # # Close the session before we exit.
    # trainer.reset()
    # trainer.close()


    # Stage 2:
    # Second, we train the entire network rather than the last layer.
    # Do the same things as the first stage.
    params = save_codes_and_config(False, second_stage_model, args.config)
    params.num_steps_per_epoch = params.num_steps_per_epoch_second_stage
    tf.set_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    get_pretrain_model(args.checkpoint,
                       os.path.join(first_stage_model, "nnet"),
                       os.path.join(second_stage_model, "nnet"))
    model_dir = os.path.join(second_stage_model, "nnet")
    with open(os.path.join(model_dir, "feature_dim"), "w") as f:
        f.write("%d\n" % dim)
    trainer = Trainer(params, second_stage_model)
    trainer.build("train",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)
    trainer.build("valid",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)
    for epoch in range(0, params.num_second_epochs):
        # Before any further training, we evaluate the performance of the current model
        valid_loss, valid_embeddings, valid_labels = trainer.valid(args.valid_dir, args.valid_spklist,
                                                                   batch_type=params.batch_type,
                                                                   output_embeddings=True)
        eer = compute_pairwise_eer(valid_embeddings, valid_labels)
        tf.logging.info("[INFO] Valid EER: %f" % eer)
        trainer.train(args.train_dir, args.train_spklist, params.second_learning_rate)
    trainer.close()


