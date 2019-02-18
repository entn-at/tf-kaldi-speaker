#!/bin/bash


. ./cmd.sh
. ./path.sh
set -e

train_nj=32
nnet_nj=32

# The virtualenv path
export TF_ENV=/home/liuyi/venv

root=/mnt/lv10/person/liuyi/fisher
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc
trials=$data/test/trials

stage=-1

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid conf local
    ln -s $kaldi_fisher/v2/utils ./
    ln -s $kaldi_fisher/v2/steps ./
    ln -s $kaldi_fisher/v2/sid ./
    ln -s $kaldi_fisher/v2/conf ./
    ln -s $kaldi_fisher/v2/local ./
    ln -s ../../voxceleb/nnet ./
fi

if [ $stage -le 0 ]; then
  local/nnet3/xvector/prepare_feats_for_egs_new.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_background $data/train_background_nosil $exp/train_background_nosil
  utils/fix_data_dir.sh $data/train_background_nosil
fi

if [ $stage -le 1 ]; then
  # have look at the length and num utts
  cp $data/train_background_nosil/utt2num_frames ./
  awk '{print $1, NF-1}' $data/train_background_nosil/spk2utt > ./spk2num
  mkdir -p $data/train_background_nosil.bak
  cp -r $data/train_background_nosil/* $data/train_background_nosil.bak

  # remove speakers with too little data
  min_len=150
  mv $data/train_background_nosil/utt2num_frames $data/train_background_nosil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/train_background_nosil/utt2num_frames.bak > $data/train_background_nosil/utt2num_frames
  utils/filter_scp.pl $data/train_background_nosil/utt2num_frames $data/train_background_nosil/utt2spk > $data/train_background_nosil/utt2spk.new
  mv $data/train_background_nosil/utt2spk.new $data/train_background_nosil/utt2spk
  utils/fix_data_dir.sh $data/train_background_nosil

  min_num_utts=5
  awk '{print $1, NF-1}' $data/train_background_nosil/spk2utt > $data/train_background_nosil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/train_background_nosil/spk2num | utils/filter_scp.pl - $data/train_background_nosil/spk2utt > $data/train_background_nosil/spk2utt.new
  mv $data/train_background_nosil/spk2utt.new $data/train_background_nosil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/train_background_nosil/spk2utt > $data/train_background_nosil/utt2spk
  utils/filter_scp.pl $data/train_background_nosil/utt2spk $data/train_background_nosil/utt2num_frames > $data/train_background_nosil/utt2num_frames.new
  mv $data/train_background_nosil/utt2num_frames.new $data/train_background_nosil/utt2num_frames
  utils/fix_data_dir.sh $data/train_background_nosil
fi

if [ $stage -le 2 ]; then
  # Split the validation set
  mkdir -p $data/train_background_nosil/valid $data/train_background_nosil/train

  num_heldout_spk=128
  num_heldout_utts_per_spk=5

  # The augmented data is similar with the not-augmented one. If an augmented version is in the valid set, it should not appear in the training data.
  # We first remove the augmented data and only sample from the original version
  sed 's/-noise//' $data/train_background_nosil/utt2spk | sed 's/-music//' | sed 's/-babble//' | sed 's/-reverb//' |\
    paste -d ' ' $data/train_background_nosil/utt2spk - | cut -d ' ' -f 1,3 > $data/train_background_nosil/utt2uniq
  utils/utt2spk_to_spk2utt.pl $data/train_background_nosil/utt2uniq > $data/train_background_nosil/uniq2utt
  cat $data/train_background_nosil/utt2spk | utils/apply_map.pl -f 1 $data/train_background_nosil/utt2uniq |\
    sort | uniq > $data/train_background_nosil/utt2spk.uniq
  utils/utt2spk_to_spk2utt.pl $data/train_background_nosil/utt2spk.uniq > $data/train_background_nosil/spk2utt.uniq
  python utils/sample_validset_spk2utt.py $num_heldout_spk $num_heldout_utts_per_spk $data/train_background_nosil/spk2utt.uniq > $data/train_background_nosil/valid/spk2utt.uniq

  # Then we find all the data that is augmented from the original version.
  cat $data/train_background_nosil/valid/spk2utt.uniq | utils/apply_map.pl -f 2- $data/train_background_nosil/uniq2utt > $data/train_background_nosil/valid/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/train_background_nosil/valid/spk2utt > $data/train_background_nosil/valid/utt2spk
  cp $data/train_background_nosil/feats.scp $data/train_background_nosil/valid
  utils/fix_data_dir.sh $data/train_background_nosil/valid

  utils/filter_scp.pl --exclude $data/train_background_nosil/valid/utt2spk $data/train_background_nosil/utt2spk > $data/train_background_nosil/train/utt2spk
  utils/utt2spk_to_spk2utt.pl $data/train_background_nosil/train/utt2spk > $data/train_background_nosil/train/spk2utt
  cp $data/train_background_nosil/feats.scp $data/train_background_nosil/train
  utils/fix_data_dir.sh $data/train_background_nosil/train

  # In the training, we need an additional file `spklist` to map the speakers to the indices.
  awk -v id=0 '{print $1, id++}' $data/train_background_nosil/train/spk2utt > $data/train_background_nosil/train/spklist
fi

if [ $stage -le 3 ]; then
  # Training a softmax network
  nnetdir=$exp/xvector_nnet_tdnn_softmax
  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax.json \
    $data/train_background_nosil/train $data/train_background_nosil/train/spklist \
    $data/train_background_nosil/valid $data/train_background_nosil/train/spklist \
    $nnetdir

  exit 1
fi


nnetdir=$exp/xvector_nnet_tdnn_softmax
checkpoint='last'

if [ $stage -le 4 ]; then
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/train_background-ivector $nnetdir/xvectors_background-ivector

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/enroll $nnetdir/xvectors_enroll

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/test $nnetdir/xvectors_test
fi

if [ $stage -le 5 ]; then
  lda_dim=150

  $train_cmd $nnetdir/xvectors_background-ivector/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_background-ivector/xvector_train_background-ivector.scp $nnetdir/xvectors_background-ivector/mean.vec || exit 1;

  $train_cmd $nnetdir/xvectors_background-ivector/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_background-ivector/xvector_train_background-ivector.scp ark:- |" \
    ark:$data/train_background-ivector/utt2spk $nnetdir/xvectors_background-ivector/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd $nnetdir/xvectors_background-ivector/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/train_background-ivector/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_background-ivector/xvector_train_background-ivector.scp ark:- | transform-vec $nnetdir/xvectors_background-ivector/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_background-ivector/plda_lda${lda_dim} || exit 1;

  $train_cmd $nnetdir/xvector_scores/log/test.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_background-ivector/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/enroll/spk2utt scp:$nnetdir/xvectors_enroll/xvector_enroll.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_background-ivector/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_background-ivector/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_background-ivector/mean.vec scp:$nnetdir/xvectors_test/xvector_test.scp ark:- | transform-vec $nnetdir/xvectors_background-ivector/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/test || exit 1;

#   python utils/recover_scores.py $trials $nnetdir/xvector_scores/test > $nnetdir/xvector_scores/fisher_test.rec
  eer=$(paste $trials $nnetdir/xvector_scores/test | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: ${eer}%"

  paste $trials $nnetdir/xvector_scores/test | awk '{print $6, $3}' > $nnetdir/xvector_scores/test.new
  grep ' target$' $nnetdir/xvector_scores/test.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/test.target
  grep ' nontarget$' $nnetdir/xvector_scores/test.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/test.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$nnetdir/xvector_scores/test.target', '$nnetdir/xvector_scores/test.nontarget', '$nnetdir/xvector_scores/test_lda_plda.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $nnetdir/xvector_scores/test.new $nnetdir/xvector_scores/test.target $nnetdir/xvector_scores/test.nontarget
  tail -n 1 $nnetdir/xvector_scores/test_lda_plda.result


  # Cosine scoring
  $train_cmd $nnetdir/xvector_scores/log/test_cos.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-mean ark:$data/enroll/spk2utt scp:$nnetdir/xvectors_enroll/xvector_enroll.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$nnetdir/xvectors_test/xvector_test.scp ark:- |" \
    $nnetdir/xvector_scores/test_cos

#  python utils/recover_scores.py $trials $nnetdir/xvector_scores/test_cos > $nnetdir/xvector_scores/test_cos.rec
  eer=$(paste $trials $nnetdir/xvector_scores/test_cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: ${eer}%"

  paste $trials $nnetdir/xvector_scores/test_cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/test_cos.new
  grep ' target$' $nnetdir/xvector_scores/test_cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/test_cos.target
  grep ' nontarget$' $nnetdir/xvector_scores/test_cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/test_cos.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$nnetdir/xvector_scores/test_cos.target', '$nnetdir/xvector_scores/test_cos.nontarget', '$nnetdir/xvector_scores/test_cos.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $nnetdir/xvector_scores/test_cos.new $nnetdir/xvector_scores/test_cos.target $nnetdir/xvector_scores/test_cos.nontarget
  tail -n 1 $nnetdir/xvector_scores/test_cos.result


  # LDA + Cosine scoring
  $train_cmd $nnetdir/xvector_scores/log/test_lda_cos.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-mean ark:$data/enroll/spk2utt scp:$nnetdir/xvectors_enroll/xvector_enroll.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_background-ivector/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_background-ivector/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_background-ivector/mean.vec scp:$nnetdir/xvectors_test/xvector_test.scp ark:- | transform-vec $nnetdir/xvectors_background-ivector/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvector_scores/test_lda_cos

#  python utils/recover_scores.py $trials $nnetdir/xvector_scores/test_lda_cos > $nnetdir/xvector_scores/test_lda_cos.rec
  eer=$(paste $trials $nnetdir/xvector_scores/test_lda_cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "EER: ${eer}%"

  paste $trials $nnetdir/xvector_scores/test_lda_cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/test_lda_cos.new
  grep ' target$' $nnetdir/xvector_scores/test_lda_cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/test_lda_cos.target
  grep ' nontarget$' $nnetdir/xvector_scores/test_lda_cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/test_lda_cos.nontarget
  cd ${KALDI_ROOT}/tools/det_score
  comm=`echo "get_eer('$nnetdir/xvector_scores/test_lda_cos.target', '$nnetdir/xvector_scores/test_lda_cos.nontarget', '$nnetdir/xvector_scores/test_lda_cos.result')"`
  echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
  cd -
  rm -f $nnetdir/xvector_scores/test_lda_cos.new $nnetdir/xvector_scores/test_lda_cos.target $nnetdir/xvector_scores/test_lda_cos.nontarget
  tail -n 1 $nnetdir/xvector_scores/test_lda_cos.result
fi




