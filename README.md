# Overview

The **tf-kaldi-speaker** implements a neural network based speaker verification system
using [Kaldi](https://github.com/kaldi-asr/kaldi) and [TensorFlow](https://github.com/tensorflow/tensorflow).

The main idea is that Kaldi can be used to do the pre- and post-processings
while TF is a better choice to build the neural network.
Compared with Kaldi nnet3, the modification of the network (e.g. adding attention, using different loss functions) using TF costs less.
Adding other features to support text-dependent speaker verification is also possible.

The purpose of the project is to make researches on neural network based speaker verification easier.
I also try to reproduce the results in my papers.

I started to develop this code when I was in Cambridge University Engineering Department (CUED) working with Prof. Mark Gales.
And I continue the development after I came back to Tsinghua University. My supervisor is Prof. Jia Liu and I'm also working with Dr. Liang He.

# Requirement

* Python: 2.7 (Update to 3.6/3.7 should be easy.)

* Kaldi: >5.5

    Since Kaldi is only used to do the pre- and post-processing, most version >5.2 works.
    Though I'm not 100% sure, I believe Kaldi with x-vector support (e.g. egs/sre16/v2) is enough.
    But if you want to run egs/voxceleb, make sure your Kaldi also contains this examples.

* Tensorflow: >1.4.0

    I write the code using TF 1.4.0 at the very beginning. Then I updated to v1.12.0.
    The future version will support TF >1.12 but I will try to make the API compatible with lower versions.
    Due to the API changes (e.g. keep_dims to keepdims in some functions), some may experience incorrect parameters.
    In that case, simply check the parameters may fix these problems.


# Methodology

The general pipeline of our framework is:

* For training:
1. Kaldi: Data preparation --> feature extraction --> training example generateion (CMVN + VAD + ...)
2. TF: Network training (training examples + nnet config)

* For test:
1. Kaldi: Data preparation --> feature extraction
2. TF: Embedding extraction
3. Kaldi: Backend classifier (Cosine/PLDA) --> performance evaluation

* Evaluate the performance:
    * MATLAB is used to compute the EER, minDCF08, minDCF10, minDCF12.
    * If you do not have MATLAB, Kaldi also provides scripts to compute the EER and minDCFs. The minDCF08 from Kaldi is 10x larger than DETware due to the computation method.

In our framework, the speaker embedding can be trained and extracted using different network architectures.
Again, the backend classifier is integrated using Kaldi.

# Features

* Entire pipeline of neural network based speaker verification.
* Both training from scratch and fine-tuning a pre-trained model are supported.
* Examples including VoxCeleb and SRE. Refer to Fisher to customized the dataset.
* Standard x-vector architecture (with minor modification).
* Angular softmax, additive margin softmax, additive angular margin softmax, triplet loss and other loss functions.
* Self attention and other attention methods (e.g. linguistic attention in text-dependent case) (*Still testing*).
* I'm now working on multi-task and text-dependent training. Hopefully they can be done in a few months.

# Usage
 * The demos for SRE and VoxCeleb 1&2 are included in egs/{sre,voxceleb}. Follow `run.sh` to go through the code.
 * The neural networks are configured using JSON files which are included in nnet_conf and the parameters are exhibited in the demos.

# Performance & Speed

I've test the code on three datasets and the results are better than the standard Kaldi recipe.
Yes, you can achieve better performance using Kaldi by carefully tuning the parameters.
But for me, I found the parameter tuning is easier using this code, especially the number of the training epochs since Kaldi ask you to decide the epochs beforehand and you cannot change the number since it generate the training examples first.

Since it only support single gpu, the speed is not very fast but acceptable in medium-scale datasets.
For VoxCeleb, the training takes about 2.5 days using Nvidia P100 and it takes 4 days for SRE.

The speed could be much faster if multi-GPUs are used. I think the multi-GPU support is not too difficult. Just add the towers and average the gradients can achieve that. The only obstacle is I don't have enough time to do that...

# Pros and cons

* Advantages
    1. Performance: The performance of our code is shown to perform better than Kaldi.
    2. Storage: There is no need to generate a *packed* egs as Kaldi to train the network. The training will load the data on the fly.
    3. Flexibility: Change the network architecture is pretty easy.
* Disadvantages
    1. Training speed: Only support single GPU at this moment. Multiple GPU can be achieved by using data parallel and tower training.
    2. Since no packed egs are generated. Multiple CPUs must be used to load the data simultaneously. This may become a shortcoming if the utterances are very long. You have to assign enough CPUs to make the loading speech fast enough and match the GPU processing speed.

# Other discussions

* I'm not sure what the best learning rate decay strategy. In this code, I provide two possible methods: using validation set and using fixed file.
The first method works well but it may take longer to train the network. The small learning rate (e.g. 1e-5) can still improve the performance while I think the learning rate decay can be faster when the learning rate is small enough.

* David Snyder and Dan Povey just released a new paper about the diarization performance using x-vector. The network in that paper is updated to more than 10 layer. You can simply change `tdnn.py` to implement the new network.
I haven't done anything to find the best network architecture. Deeper network is worth trying since we have thousands of training utterances.

# License

**Apache License, Version 2.0 (Refer to LICENCE)**

# Acknowledgements

Thanks to:

* kaldi-io-for-python: Python functions for reading kaldi data formats. Useful for rapid prototyping with python.

    <https://github.com/vesis84/kaldi-io-for-python>

* The computation resources are provided by Prof. Gales, Prof. Liu and Dr. He.

# Last ...

* UNFORTUNATELY, the code is developed under Windows. The file property cannot be maintained properly.
After downloading the code, simply run:
    ```
    find ./ -name "*.sh" | awk '{print "chmod +x "$1}' | sh
    ```
    to add the 'x' property to the .sh files.

* For cluster setup, please refer to [Kaldi](http://kaldi-asr.org/doc/queue.html) for help.
    In my case, the program is run locally.
    Modify cmd.sh and path.sh just according to standard Kaldi setup.

* If you encounter any problems or have any suggestions, please make an issue.

* If you have any extensions, feel free to create a PR.

* Details (configurations, adding network components, etc.) will be updated later.

* Contact:

    Website: <http://yiliu.org.cn>

    E-mail: liu-yi15 (at) mails (dot) tsinghua (dot) edu (dot) cn


# Please cite

I'm working on a paper describing the performance using this library. Will update soon.
