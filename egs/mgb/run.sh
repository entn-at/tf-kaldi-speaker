#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

kaldi_voxceleb=/home/dawna/mgb3/transcription/exp-yl695/software/kaldi_cpu/egs/voxceleb
feat_dir=/home/dawna/mgb3/transcription/exp-yl695/data
root_dir=/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector
mfccdir=${root_dir}/cpdaic_1.0_50/mfcc
vaddir=${root_dir}/cpdaic_1.0_50/mfcc
data=${root_dir}/cpdaic_1.0_50/data
exp=${root_dir}/cpdaic_1.0_50/exp

sub_path=decode/clust/cpdaic_1.0_50/lib/flists.nomerge

stage=5
task=dev15l

cluster_start=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.dvector.kmeans/20
cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.dvector.kmeans.ahc
result=test/result_dvector

base=/home/dawna/mgb3/transcription/exp-yl695/Snst/decode-$task
nnetdir=$exp/xvector_nnet_tdnn_softmax_sgd_1e-2
checkpoint=-1

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid conf local nnet base-diar
    ln -s $kaldi_voxceleb/v2/utils ./
    ln -s $kaldi_voxceleb/v2/steps ./
    ln -s $kaldi_voxceleb/v2/sid ./
    ln -s $kaldi_voxceleb/v2/conf ./
    ln -s $kaldi_voxceleb/v2/local ./
    ln -s ../voxceleb/nnet/ ./
    ln -s /home/dawna/mgb3/transcription/exp-yl695/base base-diar
fi

if [ $stage -le 0 ]; then
  # Get wav.scp
  mkdir -p $data/$task
  awk -F '[/-]' '{print $(NF-1)}' $feat_dir/lib/coding/${task}.fbk | sed 's/_/-/g' | paste -d ' ' $feat_dir/lib/coding/${task}.fbk - | awk '{print $3" "$1}' > $data/$task/wav.scp
  rm -f $data/$task/segments $data/$task/utt2spk
  for show in `cat $base/lib/flists.test/$task.lst`; do
    awk -F '[.=_-]' '{print $1"-"$3"-"$4"-"$7"-"$8}' $base/test/$task/${show}.1/$sub_path/${show}.scp | awk '{print $1" "$1}' >> $data/$task/utt2spk
    cp $data/$task/utt2spk $data/$task/spk2utt
    awk -F '-' '{print $0" "$1"-"$2"-"$3" "$(NF-1)/100" "$(NF)/100}' $data/$task/utt2spk |  awk '{print $1" "$3" "$4" "$5}' >> $data/$task/segments
  done
  # Do I need to modify the segments to match the actual duration?
  utils/fix_segments.sh $data/$task
  utils/fix_data_dir.sh $data/$task
fi

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $data/$task $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data/$task
fi


if [ $stage -le 2 ]; then
  # Extract the embeddings
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/$task $nnetdir/xvectors_$task
fi

if [ $stage -le 3 ]; then
  for show in `cat $base/lib/flists.test/$task.lst`; do
    mkdir -p $nnetdir/xvectors_$task/$show
    grep `echo $show | cut -d '_' -f 2` $nnetdir/xvectors_$task/xvector.scp > $nnetdir/xvectors_$task/$show/xvector.scp
    if [ -s $nnetdir/xvectors_$task/$show/xvector.scp ]; then
      python base-diar/python/create_pairwise_trials.py $nnetdir/xvectors_$task/$show/xvector.scp $nnetdir/xvectors_$task/$show/trials &
      awk -F ' ' '{print $1" 1"}' $nnetdir/xvectors_$task/$show/xvector.scp > $nnetdir/xvectors_$task/$show/num_utts.ark
    else
      echo "File $nnetdir/xvectors_$task/$show/xvector.scp is empty"
    fi
  done
  wait
fi

if [ $stage -le 4 ]; then
for show in `cat $base/lib/flists.test/$task.lst`; do
 if [ -s $nnetdir/xvectors_$task/$show/trials ]; then
   $train_cmd $nnetdir/xvectors_$task/$show/log/plda_scoring.log \
      ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:$nnetdir/xvectors_$task/$show/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_voxceleb_train/plda - |" \
      "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      $nnetdir/xvectors_$task/$show/trials $nnetdir/xvectors_$task/$show/score_plda.txt &
 fi
done
wait
fi

scoring=plda
if [ $stage -le 5 ]; then
  cd $base
  base-diar/run/step-score-eer $task $scoring $nnetdir/xvectors_$task $sub_path
  base-diar/run/step-score-cluster-eer $task $scoring normal normal $nnetdir/xvectors_$task $sub_path
  base-diar/run/step-score-cluster-eer $task $scoring center center $nnetdir/xvectors_$task $sub_path
  cd -
fi
exit 1


if [ $stage -le 6 ]; then
  cd $base
  base-diar/run/step-ahc-given-speaker-mat $task $scoring lib/flists.test/$task.speaker $nnetdir/xvectors_$task $cluster_start $cluster_result $result notlinked
  base-diar/run/ahc-full-thresh-mat -10 0.5 -5.0 $task $scoring $nnetdir/xvectors_$task $cluster_start $cluster_result $result notlinked
  cd -
fi

if [ $stage -le 7 ]; then
  # we may first use dev15 to adapt the PLDA and see what's going on
  base_dev15=/home/dawna/mgb3/transcription/exp-yl695/Snst/decode-dev15
  mkdir -p $data/dev15
  awk -F '[/-]' '{print $(NF-1)}' $feat_dir/lib/coding/dev15.fbk | sed 's/_/-/g' | paste -d ' ' $feat_dir/lib/coding/dev15.fbk - | awk '{print $3" "$1}' > $data/dev15/wav.scp
  rm -f $data/dev15/segments $data/dev15/utt2spk
  for show in `cat $base_dev15/lib/flists.test/dev15.lst`; do
    awk -F '[.=_-]' '{print $1"-"$3"-"$4"-"$7"-"$8}' $base_dev15/test/dev15/${show}.1/$sub_path/${show}.scp | awk '{print $1" "$1}' >> $data/dev15/utt2spk
    cp $data/dev15/utt2spk $data/dev15/spk2utt
    awk -F '-' '{print $0" "$1"-"$2"-"$3" "$(NF-1)/100" "$(NF)/100}' $data/dev15/utt2spk |  awk '{print $1" "$3" "$4" "$5}' >> $data/dev15/segments
  done
  # Do I need to modify the segments to match the actual duration?
  utils/fix_segments.sh $data/dev15
  utils/fix_data_dir.sh $data/dev15
fi

if [ $stage -le 8 ]; then
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $data/dev15 $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data/dev15
fi

if [ $stage -le 9 ]; then
  # Extract the embeddings
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/dev15 $nnetdir/xvectors_dev15
fi

if [ $stage -le 10 ]; then
  $train_cmd $nnetdir/xvectors_dev15/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_dev15/xvector.scp \
      $nnetdir/xvectors_dev15/mean.vec || exit 1;
  $train_cmd $nnetdir/xvectors_dev15/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $nnetdir/xvectors_voxceleb_train/plda \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_dev15/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_dev15/plda_adapt || exit 1;
fi

if [ $stage -le 11 ]; then
  for show in `cat $base/lib/flists.test/$task.lst`; do
    if [ -s $nnetdir/xvectors_$task/$show/trials ]; then
      $train_cmd $nnetdir/xvectors_$task/$show/log/plda_adapt_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:$nnetdir/xvectors_$task/$show/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_dev15/plda_adapt - |" \
        "ark:ivector-subtract-global-mean $nnetdir/xvectors_dev15/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $nnetdir/xvectors_dev15/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $nnetdir/xvectors_$task/$show/trials $nnetdir/xvectors_$task/$show/score_plda_adapt.txt &
    fi
  done
  wait
fi

scoring=plda_adapt
if [ $stage -le 12 ]; then
  cd $base
  base-diar/run/step-score-eer $task $scoring $nnetdir/xvectors_$task $sub_path
  base-diar/run/step-score-cluster-eer $task $scoring normal normal $nnetdir/xvectors_$task $sub_path
  base-diar/run/step-score-cluster-eer $task $scoring center center $nnetdir/xvectors_$task $sub_path
  cd -
fi

if [ $stage -le 13 ]; then
  cd $base
  base-diar/run/step-ahc-given-speaker-mat $task $scoring lib/flists.test/$task.speaker $nnetdir/xvectors_$task $cluster_start $cluster_result $result notlinked
  base-diar/run/ahc-full-thresh-mat -13 0.5 -8.0 $task $scoring $nnetdir/xvectors_$task $cluster_start $cluster_result $result notlinked
  cd -
fi
