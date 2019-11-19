#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2019   Yi Liu. Modified to support network training using TensorFlow
# Apache 2.0.
#
# Change to the official trainging/test list. The models can be compared with other works, rather than the Kaldi result.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

# make sure to modify "cmd.sh" and "path.sh", change the KALDI_ROOT to the correct directory
. ./cmd.sh
. ./path.sh
set -e

root=/home/heliang05/liuyi/voxceleb.official
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc

stage=5

# The kaldi voxceleb egs directory
kaldi_voxceleb=/home/heliang05/liuyi/software/kaldi_gpu/egs/voxceleb

voxceleb1_trials=$data/voxceleb_test/trials
voxceleb1_root=/home/heliang05/liuyi/data/voxceleb/voxceleb1
voxceleb2_root=/home/heliang05/liuyi/data/voxceleb/voxceleb2
musan_root=/home/heliang05/liuyi/data/musan
rirs_root=/home/heliang05/liuyi/data/RIRS_NOISES

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid conf local
    ln -s $kaldi_voxceleb/v2/utils ./
    ln -s $kaldi_voxceleb/v2/steps ./
    ln -s $kaldi_voxceleb/v2/sid ./
    ln -s $kaldi_voxceleb/v2/conf ./
    ln -s $kaldi_voxceleb/v2/local ./

    ln -s ../../voxceleb/v1/nnet ./
    exit 1
fi

if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev $data/voxceleb2_train
  local/make_voxceleb1.pl $voxceleb1_root $data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in voxceleb2_train voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      $data/${name} $exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      $data/${name} $exp/make_vad $vaddir
    utils/fix_data_dir.sh $data/${name}
  done

  cp -r $data/voxceleb2_train $data/voxceleb_train
  cp -r $data/voxceleb1_test $data/voxceleb_test
  exit 1
fi

# Without augmentation

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 2 ]; then
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    $data/voxceleb_train $data/voxceleb_train_no_sil $exp/voxceleb_train_no_sil
  utils/fix_data_dir.sh $data/voxceleb_train_no_sil
  cp -r $data/voxceleb_train_no_sil $data/voxceleb_train_no_sil.bak
  exit 1
fi

if [ $stage -le 3 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv $data/voxceleb_train_no_sil/utt2num_frames $data/voxceleb_train_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/voxceleb_train_no_sil/utt2num_frames.bak > $data/voxceleb_train_no_sil/utt2num_frames
  utils/filter_scp.pl $data/voxceleb_train_no_sil/utt2num_frames $data/voxceleb_train_no_sil/utt2spk > $data/voxceleb_train_no_sil/utt2spk.new
  mv $data/voxceleb_train_no_sil/utt2spk.new $data/voxceleb_train_no_sil/utt2spk
  utils/fix_data_dir.sh $data/voxceleb_train_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' $data/voxceleb_train_no_sil/spk2utt > $data/voxceleb_train_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/voxceleb_train_no_sil/spk2num | utils/filter_scp.pl - $data/voxceleb_train_no_sil/spk2utt > $data/voxceleb_train_no_sil/spk2utt.new
  mv $data/voxceleb_train_no_sil/spk2utt.new $data/voxceleb_train_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/voxceleb_train_no_sil/spk2utt > $data/voxceleb_train_no_sil/utt2spk

  utils/filter_scp.pl $data/voxceleb_train_no_sil/utt2spk $data/voxceleb_train_no_sil/utt2num_frames > $data/voxceleb_train_no_sil/utt2num_frames.new
  mv $data/voxceleb_train_no_sil/utt2num_frames.new $data/voxceleb_train_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh $data/voxceleb_train_no_sil
  exit 1
fi

if [ $stage -le 4 ]; then
  # Split the validation set
  num_heldout_spks=200
  num_heldout_utts_per_spk=5
  mkdir -p $data/voxceleb_train_no_sil/train/ $data/voxceleb_train_no_sil/valid/

  sed 's/-noise//' $data/voxceleb_train_no_sil/utt2spk | sed 's/-music//' | sed 's/-babble//' | sed 's/-reverb//' |\
    paste -d ' ' $data/voxceleb_train_no_sil/utt2spk - | cut -d ' ' -f 1,3 > $data/voxceleb_train_no_sil/utt2uniq

  utils/utt2spk_to_spk2utt.pl $data/voxceleb_train_no_sil/utt2uniq > $data/voxceleb_train_no_sil/uniq2utt
  cat $data/voxceleb_train_no_sil/utt2spk | utils/apply_map.pl -f 1 $data/voxceleb_train_no_sil/utt2uniq |\
    sort | uniq > $data/voxceleb_train_no_sil/utt2spk.uniq

  utils/utt2spk_to_spk2utt.pl $data/voxceleb_train_no_sil/utt2spk.uniq > $data/voxceleb_train_no_sil/spk2utt.uniq
  python $TF_KALDI_ROOT/misc/tools/sample_validset_spk2utt.py $num_heldout_spks $num_heldout_utts_per_spk $data/voxceleb_train_no_sil/spk2utt.uniq > $data/voxceleb_train_no_sil/valid/spk2utt.uniq

  cat $data/voxceleb_train_no_sil/valid/spk2utt.uniq | utils/apply_map.pl -f 2- $data/voxceleb_train_no_sil/uniq2utt > $data/voxceleb_train_no_sil/valid/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/voxceleb_train_no_sil/valid/spk2utt > $data/voxceleb_train_no_sil/valid/utt2spk
  cp $data/voxceleb_train_no_sil/feats.scp $data/voxceleb_train_no_sil/valid
  utils/filter_scp.pl $data/voxceleb_train_no_sil/valid/utt2spk $data/voxceleb_train_no_sil/utt2num_frames > $data/voxceleb_train_no_sil/valid/utt2num_frames
  utils/fix_data_dir.sh $data/voxceleb_train_no_sil/valid

  utils/filter_scp.pl --exclude $data/voxceleb_train_no_sil/valid/utt2spk $data/voxceleb_train_no_sil/utt2spk > $data/voxceleb_train_no_sil/train/utt2spk
  utils/utt2spk_to_spk2utt.pl $data/voxceleb_train_no_sil/train/utt2spk > $data/voxceleb_train_no_sil/train/spk2utt
  cp $data/voxceleb_train_no_sil/feats.scp $data/voxceleb_train_no_sil/train
  utils/filter_scp.pl $data/voxceleb_train_no_sil/train/utt2spk $data/voxceleb_train_no_sil/utt2num_frames > $data/voxceleb_train_no_sil/train/utt2num_frames
  utils/fix_data_dir.sh $data/voxceleb_train_no_sil/train

  awk -v id=0 '{print $1, id++}' $data/voxceleb_train_no_sil/train/spk2utt > $data/voxceleb_train_no_sil/train/spklist
exit 1
fi


if [ $stage -le 5 ]; then
  # Training a softmax network
  nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2
  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_1e-2.json \
    $data/voxceleb_train_no_sil/train $data/voxceleb_train_no_sil/train/spklist \
    $data/voxceleb_train_no_sil/valid $data/voxceleb_train_no_sil/train/spklist \
    $nnetdir

exit 1
fi


nnetdir=$exp/
checkpoint='last'

if [ $stage -le 6 ]; then
  # Extract the embeddings
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 80 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/voxceleb_train $nnetdir/xvectors_voxceleb_train

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/voxceleb_test $nnetdir/xvectors_voxceleb_test
fi

if [ $stage -le 7 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnetdir/xvectors_voxceleb_train/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_voxceleb_train/xvector.scp \
    $nnetdir/xvectors_voxceleb_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $nnetdir/xvectors_voxceleb_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_voxceleb_train/xvector.scp ark:- |" \
    ark:$data/voxceleb_train/utt2spk $nnetdir/xvectors_voxceleb_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnetdir/xvectors_voxceleb_train/log/plda.log \
    ivector-compute-plda ark:$data/voxceleb_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_voxceleb_train/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnetdir/xvectors_voxceleb_train/plda || exit 1;
fi

if [ $stage -le 8 ]; then
  $train_cmd $nnetdir/scores/log/voxceleb_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_voxceleb_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $nnetdir/scores/scores_voxceleb_test.plda || exit 1;

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.plda) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --c-miss 10 --p-target 0.01 $nnetdir/scores/scores_voxceleb_test.plda $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/scores_voxceleb_test.plda $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"

  # Use DETware provided by NIST. It requires MATLAB to compute the DET and DCF.
  paste -d ' ' $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.plda | grep ' target ' | awk '{print $NF}' > $nnetdir/scores/scores_voxceleb_test.plda.target
  paste -d ' ' $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.plda | grep ' nontarget ' | awk '{print $NF}' > $nnetdir/scores/scores_voxceleb_test.plda.nontarget
  comm=`echo "addpath('../../../misc/DETware_v2.1');Get_DCF('$nnetdir/scores/scores_voxceleb_test.plda.target', '$nnetdir/scores/scores_voxceleb_test.plda.nontarget', '$nnetdir/scores/scores_voxceleb_test.plda.result');"`
  echo "$comm"| matlab -nodesktop > /dev/null
  tail -n 1 $nnetdir/scores/scores_voxceleb_test.plda.result
  exit 1
fi



if [ $stage -le 9 ]; then
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/voxceleb_test $nnetdir/xvectors_voxceleb_test
fi

if [ $stage -le 10 ]; then
  # Cosine similarity
  mkdir -p $nnetdir/scores
  cat $voxceleb1_trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
      "ark:ivector-normalize-length scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- |" \
      "ark:ivector-normalize-length scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- |" \
      $nnetdir/scores/scores_voxceleb_test.cos

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.cos) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --c-miss 10 --p-target 0.01 $nnetdir/scores/scores_voxceleb_test.cos $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/scores_voxceleb_test.cos $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"

  # Comment the following lines if you do not have matlab.
  paste -d ' ' $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.cos | grep ' target ' | awk '{print $NF}' > $nnetdir/scores/scores_voxceleb_test.cos.target
  paste -d ' ' $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.cos | grep ' nontarget ' | awk '{print $NF}' > $nnetdir/scores/scores_voxceleb_test.cos.nontarget
  comm=`echo "addpath('../../misc/DETware_v2.1');Get_DCF('$nnetdir/scores/scores_voxceleb_test.cos.target', '$nnetdir/scores/scores_voxceleb_test.cos.nontarget', '$nnetdir/scores/scores_voxceleb_test.cos.result');"`
  echo "$comm"| matlab -nodesktop > /dev/null
  tail -n 1 $nnetdir/scores/scores_voxceleb_test.cos.result
fi
