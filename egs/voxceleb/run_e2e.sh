#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
#             2018   Yi Liu. Modified to support network training using TensorFlow
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

# make sure to modify "cmd.sh" and "path.sh", change the KALDI_ROOT to the correct directory
. ./cmd.sh
. ./path.sh
set -e

source activate tf

data=/home/dawna/mgb3/diarization/imports/data/mfc30/data
data2=/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector/cpdaic_1.0_50/data
exp=/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector/cpdaic_1.0_50/exp
mfccdir=/home/dawna/mgb3/diarization/imports/data/mfc30/mfcc
vaddir=/home/dawna/mgb3/diarization/imports/data/mfc30/mfcc

stage=1

# The kaldi voxceleb egs directory
kaldi_voxceleb=/home/dawna/mgb3/transcription/exp-yl695/software/kaldi_cpu/egs/voxceleb

voxceleb1_trials=/home/dawna/mgb3/diarization/imports/data/mfc30/data/voxceleb1_test/trials
voxceleb1_root=/home/dawna/mgb3/diarization/imports/voxceleb/voxceleb1
voxceleb2_root=/home/dawna/mgb3/diarization/imports/voxceleb/voxceleb2
musan_root=/home/dawna/mgb3/diarization/imports/musan
rirs_root=/home/dawna/mgb3/diarization/imports/RIRS_NOISES

if [ $stage -le 0 ]; then
#nnetdir=$exp/xvector_nnet_ge2e_softmax
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --continue-training false nnet_conf/tdnn_ge2e.json \
#    $data2/voxceleb_train_combined_no_sil/train $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $data2/voxceleb_train_combined_no_sil/end2end_valid $data2/voxceleb_train_combined_no_sil/end2end_valid/spklist \
#    $nnetdir


exit 1
echo
fi

nnetdir=$exp/xvector_nnet_tdnn_softmax_to_ge2e_softmax
checkpoint=-1

if [ $stage -le 1 ]; then
  # Extract the embeddings
#  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 200 --use-gpu false --checkpoint $checkpoint --stage 0 \
#    --chunk-size 10000 --normalize true \
#    $nnetdir $data2/voxceleb_train $nnetdir/xvectors_voxceleb_train

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize true \
    $nnetdir $data2/voxceleb_test $nnetdir/xvectors_voxceleb_test
fi

if [ $stage -le 2 ]; then
  # Cosine similarity
  mkdir -p $nnetdir/scores
#  cat $voxceleb1_trials | awk '{print $1, $2}' | \
#    ivector-compute-dot-products - \
#      "ark:ivector-normalize-length scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- |" \
#      "ark:ivector-normalize-length scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- |" \
#      $nnetdir/scores/scores_voxceleb_test.cos

  cat $voxceleb1_trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
      scp:$nnetdir/xvectors_voxceleb_test/xvector.scp \
      scp:$nnetdir/xvectors_voxceleb_test/xvector.scp \
      $nnetdir/scores/scores_voxceleb_test.cos

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.cos) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnetdir/scores/scores_voxceleb_test.cos $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/scores_voxceleb_test.cos $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
exit 1


if [ $stage -le 3 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnetdir/xvectors_voxceleb_train/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_voxceleb_train/xvector.scp \
    $nnetdir/xvectors_voxceleb_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $nnetdir/xvectors_voxceleb_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_voxceleb_train/xvector.scp ark:- |" \
    ark:$data2/voxceleb_train/utt2spk $nnetdir/xvectors_voxceleb_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnetdir/xvectors_voxceleb_train/log/plda.log \
    ivector-compute-plda ark:$data2/voxceleb_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_voxceleb_train/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnetdir/xvectors_voxceleb_train/plda || exit 1;
fi

if [ $stage -le 4 ]; then
  $train_cmd $nnetdir/scores/log/voxceleb_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_voxceleb_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $nnetdir/scores/scores_voxceleb_test.plda || exit 1;

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.plda) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnetdir/scores/scores_voxceleb_test.plda $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/scores_voxceleb_test.plda $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.128%
  # minDCF(p-target=0.01): 0.3258
  # minDCF(p-target=0.001): 0.5003
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
fi
