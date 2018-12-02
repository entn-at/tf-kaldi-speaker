import tensorflow as tf
from model.pooling import statistics_pooling
from collections import OrderedDict
from model.common import l2_normalize
from model.loss import ge2e_loss, test_loss
from misc.utils import ParamsPlain
from model.loss import softmax, ge2e_loss, triplet_loss, test_loss
from dataset.data_loader import KaldiDataRandomQueue


def test_network(features, params, is_training=None, reuse_variables=None):
    """Test a new network. """
    endpoints = OrderedDict()
    with tf.variable_scope("test", reuse=reuse_variables):
        # Convert to [b, 1, l, d]
        features = tf.expand_dims(features, 1)

        # Layer 1: [-2,-1,0,1,2] --> [b, 1, l-4, 512]
        # conv2d + batchnorm + relu
        features = tf.layers.conv2d(features,
                                512,
                                (1, 5),
                                activation=None,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                name='tdnn1_conv')
        endpoints["tdnn1_conv"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn1_bn")
        endpoints["tdnn1_bn"] = features
        features = tf.nn.relu(features, name='tdnn1_relu')
        endpoints["tdnn1_relu"] = features

        # Convert to [b, l, 512]
        features = tf.squeeze(features, axis=1)

        # Layer 5: [b, l, x]
        if "num_nodes_pooling_layer" not in params.dict:
            # The default number of nodes before pooling
            params.dict["num_nodes_pooling_layer"] = 1500

        features = tf.layers.dense(features,
                                   params.num_nodes_pooling_layer,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="tdnn5_dense")
        endpoints["tdnn5_dense"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn5_bn")
        endpoints["tdnn5_bn"] = features
        features = tf.nn.relu(features, name='tdnn5_relu')
        endpoints["tdnn5_relu"] = features

        # Statistics pooling
        # [b, l, 1500] --> [b, 1500]
        if params.pooling_type == "statistics_pooling":
            features = statistics_pooling(features)
        else:
            raise NotImplementedError("Not implement %s pooling" % params.poolingtype)
        endpoints['pooling'] = features

        # Layer 7: [b, x]
        if "num_nodes_last_layer" not in params.dict:
            # The default number of nodes in the last layer
            params.dict["num_nodes_last_layer"] = 512
        features = tf.layers.dense(features,
                                   params.num_nodes_last_layer,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='tdnn7_dense')
        endpoints['tdnn7_dense'] = features

        features = l2_normalize(features)
        endpoints['embeddings'] = features

    return features, endpoints


if __name__ == "__main__":
    data = '/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector/cpdaic_1.0_50/data/voxceleb_train_combined_no_sil/train/'
    spklist ='/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector/cpdaic_1.0_50/data/voxceleb_train_combined_no_sil/train/spklist'
    params = ParamsPlain()
    params.dict["num_speakers_per_batch"] = 64
    params.dict["num_segments_per_speaker"] = 10
    params.dict["init_end2end_w"] = 10
    params.dict["init_end2end_b"] = -5
    params.dict["ge2e_loss_type"] = "softmax"
    params.dict["weight_l2_regularizer"] = 1e-5
    params.dict["batchnorm_momentum"] = 0.999
    params.dict["pooling_type"] = "statistics_pooling"

    features = tf.placeholder(tf.float32, shape=[None, None, 30], name="train_features")
    labels = tf.placeholder(tf.int32, shape=[None, ], name="train_labels")
    _, endpoints = test_network(features, params, True, False)
    loss1, loss_ref = test_loss(endpoints['embeddings'], labels, None, params, True, False)
    loss2 = ge2e_loss(endpoints['tdnn7_dense'], labels, None, params, True, True)

    data_loader = KaldiDataRandomQueue(data, spklist,
                                       num_parallel=4,
                                       max_qsize=10,
                                       num_speakers=params.num_speakers_per_batch,
                                       num_segments=params.num_segments_per_speaker,
                                       min_len=200,
                                       max_len=400,
                                       shuffle=True)
    data_loader.start()
    features_val, labels_val = data_loader.fetch()
    data_loader.stop()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [endpoints_val, loss1_val, loss2_val, loss_ref_val] = sess.run([endpoints, loss1, loss2, loss_ref], feed_dict={features: features_val,
                                                                               labels: labels_val})

        from model.test_utils import compute_ge2e_loss
        loss_np = compute_ge2e_loss(endpoints_val["tdnn7_dense"], labels_val, 10, -5, params.ge2e_loss_type)
        print(loss_np)
