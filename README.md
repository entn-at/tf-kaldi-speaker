# Overview

The **tf-kaldi-speaker** implements a neural network based speaker verification system
using [Kaldi](https://github.com/kaldi-asr/kaldi) and [TensorFlow](https://github.com/tensorflow/tensorflow).

The main idea is that Kaldi can be used to do the pre- and post-processings
while TF is a better choice to build the neural network.
Compared with Kaldi nnet3, the modification of the network (e.g. adding attention, using different loss functions) using TF costs less.
Adding other features to support text-dependent speaker verification is also possible.

The purpose of the project is to make researches on neural network based speaker verification easier.
I also try to reproduce the results in my papers.

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

* Entire pipeline of neural network speaker verification.
* Examples including VoxCeleb.
* Standard x-vector architecture (with minor modification).
* Angular softmax, additive margin softmax, additive angular margin softmax.
* Self attention and other attention methods (e.g. linguistic attention in text-dependent case).
* TODO: Add egs for SRE and Fiser.


# License

**The code is under Apache 2.0. (refer to "LICENSE")**

# Acknowledgements

Thanks to:

* kaldi-io-for-python: Python functions for reading kaldi data formats. Useful for rapid prototyping with python.

    <https://github.com/vesis84/kaldi-io-for-python>

# Last ...

* UNFORTUNATELY, the code is developed under Windows. The property of files cannot be maintained properly.
After downloading the code, simply run:
    ```
    find ./ -name "*.sh" | awk '{print "chmod +x "$1}' | sh
    ```
    to add the 'x' property to the .sh files.

* For cluster setup, please refer to [Kaldi](http://kaldi-asr.org/doc/queue.html) for help.
    In my case, the program is run locally.
    Modify cmd.sh and path.sh just according to standard Kaldi setup.

* **The code will be updated later to support attention, joint training with acoustic model, etc.
    Refer to `CHANGELOG.md` for more details.**

* If you encounter any problems or have any suggestions, please make an issue.

* If you have any extensions, feel free to create a PR.

* My website: <http://yiliu.org.cn>

    E-mail: liu-yi15 (at) mails (dot) tsinghua (dot) edu (dot) cn
