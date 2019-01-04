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

stage=7

# The kaldi voxceleb egs directory
kaldi_voxceleb=/home/dawna/mgb3/transcription/exp-yl695/software/kaldi_cpu/egs/voxceleb

voxceleb1_trials=/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector/cpdaic_1.0_50/data/voxceleb_test/trials
voxceleb1_root=/home/dawna/mgb3/diarization/imports/voxceleb/voxceleb1
voxceleb2_root=/home/dawna/mgb3/diarization/imports/voxceleb/voxceleb2
musan_root=/home/dawna/mgb3/diarization/imports/musan
rirs_root=/home/dawna/mgb3/diarization/imports/RIRS_NOISES

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid conf local
    ln -s $kaldi_voxceleb/v2/utils ./
    ln -s $kaldi_voxceleb/v2/steps ./
    ln -s $kaldi_voxceleb/v2/sid ./
    ln -s $kaldi_voxceleb/v2/conf ./
    ln -s $kaldi_voxceleb/v2/local ./
fi

if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev $data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test $data/voxceleb2_test
  # This script reates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1.pl $voxceleb1_root $data
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,351 speakers and 1,277,503 utterances.
  utils/combine_data.sh $data/voxceleb_train $data/voxceleb2_train $data/voxceleb2_test $data/voxceleb1_train
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in voxceleb_train voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      $data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      $data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh $data/${name}
  done
fi

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data/voxceleb_train/utt2num_frames > $data/voxceleb_train/reco2dur

  # Make sure you already have the RIRS_NOISES dataset
#  # Make a version with reverberated speech
#  rvb_opts=()
#  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
#  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    $data/voxceleb_train $data/voxceleb_train_reverb
  cp data/voxceleb_train/vad.scp $data/voxceleb_train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" $data/voxceleb_train_reverb $data/voxceleb_train_reverb.new
  rm -rf $data/voxceleb_train_reverb
  mv $data/voxceleb_train_reverb.new $data/voxceleb_train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh $musan_root $data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh $data/musan_${name}
    mv $data/musan_${name}/utt2dur $data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$data/musan_noise" $data/voxceleb_train $data/voxceleb_train_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$data/musan_music" $data/voxceleb_train $data/voxceleb_train_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$data/musan_speech" $data/voxceleb_train $data/voxceleb_train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh $data/voxceleb_train_aug $data/voxceleb_train_reverb $data/voxceleb_train_noise $data/voxceleb_train_music $data/voxceleb_train_babble
fi

if [ $stage -le 3 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh $data/voxceleb_train_aug 1000000 $data/voxceleb_train_aug_1m
  utils/fix_data_dir.sh $data/voxceleb_train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $data/voxceleb_train_aug_1m $exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh $data/voxceleb_train_combined $data/voxceleb_train_aug_1m $data/voxceleb_train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    $data/voxceleb_train_combined $data2/voxceleb_train_combined_no_sil $exp/voxceleb_train_combined_no_sil
  utils/fix_data_dir.sh $data2/voxceleb_train_combined_no_sil
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv $data2/voxceleb_train_combined_no_sil/utt2num_frames $data2/voxceleb_train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data2/voxceleb_train_combined_no_sil/utt2num_frames.bak > $data2/voxceleb_train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl $data2/voxceleb_train_combined_no_sil/utt2num_frames $data2/voxceleb_train_combined_no_sil/utt2spk > $data2/voxceleb_train_combined_no_sil/utt2spk.new
  mv $data2/voxceleb_train_combined_no_sil/utt2spk.new $data2/voxceleb_train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh $data2/voxceleb_train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' $data2/voxceleb_train_combined_no_sil/spk2utt > $data2/voxceleb_train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data2/voxceleb_train_combined_no_sil/spk2num | utils/filter_scp.pl - $data2/voxceleb_train_combined_no_sil/spk2utt > $data2/voxceleb_train_combined_no_sil/spk2utt.new
  mv $data2/voxceleb_train_combined_no_sil/spk2utt.new $data2/voxceleb_train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data2/voxceleb_train_combined_no_sil/spk2utt > $data2/voxceleb_train_combined_no_sil/utt2spk

  utils/filter_scp.pl $data2/voxceleb_train_combined_no_sil/utt2spk $data2/voxceleb_train_combined_no_sil/utt2num_frames > $data2/voxceleb_train_combined_no_sil/utt2num_frames.new
  mv $data2/voxceleb_train_combined_no_sil/utt2num_frames.new $data2/voxceleb_train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh $data2/voxceleb_train_combined_no_sil
fi

if [ $stage -le 6 ]; then
  # Split the validation set
  # There are 2 validation sets, 1 for softmax-like loss function and 1 for end2end loss.
  # The speakers are the same with the training set for softmax-like loss
  # and the speakers are different for end2end loss.

  # Get the validation set.
  num_heldout_spks=64
  num_heldout_utts=1000
  mkdir -p $data2/voxceleb_train_combined_no_sil/softmax_valid/ $data2/voxceleb_train_combined_no_sil/end2end_valid/ $data2/voxceleb_train_combined_no_sil/train

  echo "$0: Preparing end2end loss validation lists"
  end2end_valid_dir=$data2/voxceleb_train_combined_no_sil/end2end_valid/
  awk '{print (NF-1)" "$1}' $data2/voxceleb_train_combined_no_sil/spk2utt | sort -nr | awk '{print $2" "$1}' | head -1000 > $end2end_valid_dir/valid_candidate
  utils/filter_scp.pl $end2end_valid_dir/valid_candidate $data2/voxceleb_train_combined_no_sil/spk2utt | utils/shuffle_list.pl | head -$num_heldout_spks > $end2end_valid_dir/spk2utt || exit 1
  utils/spk2utt_to_utt2spk.pl $end2end_valid_dir/spk2utt > $end2end_valid_dir/utt2spk
  cp $data2/voxceleb_train_combined_no_sil/feats.scp $end2end_valid_dir
  utils/fix_data_dir.sh $end2end_valid_dir

  echo "$0: Preparing softmax loss validation lists"
  train_dir=$data2/voxceleb_train_combined_no_sil/train/
  softmax_valid_dir=$data2/voxceleb_train_combined_no_sil/softmax_valid/
  utils/filter_scp.pl --exclude $end2end_valid_dir/spk2utt $data2/voxceleb_train_combined_no_sil/spk2utt > $train_dir/spk2utt
  utils/spk2utt_to_utt2spk.pl $train_dir/spk2utt > $train_dir/utt2spk
  cp $data2/voxceleb_train_combined_no_sil/feats.scp $train_dir
  utils/filter_scp.pl $train_dir/utt2spk $data2/voxceleb_train_combined_no_sil/utt2num_frames > $train_dir/utt2num_frames
  utils/fix_data_dir.sh $train_dir

  awk '{print $2" "$1}' $train_dir/utt2num_frames | sort -nr | awk '{print $2" "$1}' | head -30000 > $softmax_valid_dir/valid_candidate
  utils/filter_scp.pl $softmax_valid_dir/valid_candidate $train_dir/utt2spk | utils/shuffle_list.pl | head -$num_heldout_utts > $softmax_valid_dir/utt2spk || exit 1;
  utils/utt2spk_to_spk2utt.pl $softmax_valid_dir/utt2spk > $softmax_valid_dir/spk2utt
  cp $data2/voxceleb_train_combined_no_sil/feats.scp $softmax_valid_dir
  utils/fix_data_dir.sh $softmax_valid_dir

  utils/filter_scp.pl --exclude $softmax_valid_dir/utt2spk $train_dir/utt2spk > $train_dir/utt2spk.new
  mv $train_dir/utt2spk.new $train_dir/utt2spk
  utils/fix_data_dir.sh $train_dir

  # In the training, we need an additional file `spklist` to map the speakers to the indices.
  # This file should be generated manually.
  awk -v id=0 '{print $1, id++}' $train_dir/spk2utt > $train_dir/spklist
  awk -v id=0 '{print $1, id++}' $end2end_valid_dir/spk2utt > $end2end_valid_dir/spklist
fi

if [ $stage -le 7 ]; then
## Training a softmax network
#nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --continue-training false nnet_conf/tdnn_softmax_1e-2.json \
#    $data2/voxceleb_train_combined_no_sil/train $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $data2/voxceleb_train_combined_no_sil/softmax_valid $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir


#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_fn_s20_m4
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --continue-training false nnet_conf/tdnn_asoftmax_fn_1e-2_s20_m4.json \
#    $data2/voxceleb_train_combined_no_sil/train $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $data2/voxceleb_train_combined_no_sil/softmax_valid $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir

#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_fn_s20_linear_m4
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --continue-training false nnet_conf/tdnn_asoftmax_linear_fn_1e-2_s20_m4.json \
#    $data2/voxceleb_train_combined_no_sil/train $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $data2/voxceleb_train_combined_no_sil/softmax_valid $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir

#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_1e-2_m2_long
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --continue-training false nnet_conf/tdnn_asoftmax_1e-2_m2_long.json \
#    $data2/voxceleb_train_combined_no_sil/train $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $data2/voxceleb_train_combined_no_sil/softmax_valid $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir

#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_1e-2_m4_long
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --continue-training false nnet_conf/tdnn_asoftmax_1e-2_m4_long.json \
#    $data2/voxceleb_train_combined_no_sil/train $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $data2/voxceleb_train_combined_no_sil/softmax_valid $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir

nnetdir=$exp/xvector_nnet_tdnn_asoftmax_fn_s20_linear_m2
nnet/run_train_nnet.sh --cmd "$cuda_cmd" --continue-training false nnet_conf/tdnn_asoftmax_linear_fn_1e-2_s20_m2.json \
    $data2/voxceleb_train_combined_no_sil/train $data2/voxceleb_train_combined_no_sil/train/spklist \
    $data2/voxceleb_train_combined_no_sil/softmax_valid $data2/voxceleb_train_combined_no_sil/train/spklist \
    $nnetdir

#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_fn_s20_m2
#nnet/run_train_nnet.sh --cmd "$cuda_cmd" --continue-training false nnet_conf/tdnn_asoftmax_fn_1e-2_s20_m2.json \
#    $data2/voxceleb_train_combined_no_sil/train $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $data2/voxceleb_train_combined_no_sil/softmax_valid $data2/voxceleb_train_combined_no_sil/train/spklist \
#    $nnetdir

exit 1
echo
fi


nnetdir=$exp/xvector_nnet_tdnn_asoftmax_1e-2_m2_long
checkpoint=-1

if [ $stage -le 8 ]; then
  # Extract the embeddings
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 60 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data2/voxceleb_train $nnetdir/xvectors_voxceleb_train

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data2/voxceleb_test $nnetdir/xvectors_voxceleb_test
fi

if [ $stage -le 9 ]; then
  # Cosine similarity
  mkdir -p $nnetdir/scores
  cat $voxceleb1_trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
      "ark:ivector-normalize-length scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- |" \
      "ark:ivector-normalize-length scp:$nnetdir/xvectors_voxceleb_test/xvector.scp ark:- |" \
      $nnetdir/scores/scores_voxceleb_test.cos

  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.cos) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnetdir/scores/scores_voxceleb_test.cos $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/scores_voxceleb_test.cos $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi


if [ $stage -le 10 ]; then
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

if [ $stage -le 11 ]; then
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
exit 1

if [ $stage -le 12 ]; then


  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $nnetdir/scores/scores_voxceleb_test.plda) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnetdir/scores/scores_voxceleb_test.plda $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/scores_voxceleb_test.plda $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

