#!/bin/bash

. ./cmd.sh
. ./path.sh

kaldi_voxceleb=/home/dawna/mgb3/transcription/exp-yl695/software/kaldi_cpu/egs/voxceleb
feat_dir=/home/dawna/mgb3/transcription/exp-yl695/data
root_dir=/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector
mfccdir=${root_dir}/cpdaic_1.0_50/mfcc
vaddir=${root_dir}/cpdaic_1.0_50/mfcc
data=${root_dir}/cpdaic_1.0_50/data
exp=${root_dir}/cpdaic_1.0_50/exp

stage=28
task=dev15l

base=/home/dawna/mgb3/transcription/exp-yl695/Snst/decode-$task
base_dev15=/home/dawna/mgb3/transcription/exp-yl695/Snst/decode-dev15
base_train=/home/dawna/mgb3/transcription/exp-yl695/Snst/decode-train

#from_org=-ORG

scoring=plda

nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2_m0.9
checkpoint=-1
sub_path=decode/clust/cpdaic_1.0_50/lib/flists.nomerge
purify_cluster=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.kmeans.softmax_plda
cluster_start=$purify_cluster/20
cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.ahc.softmax_plda
result=test/result_dvector_softmax_plda

adapt_nnetdir=$nnetdir/adapt/stage_two
data_adapt=$data/softmax_plda
adapt_checkpoint=5000
adapt_purify_cluster=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.kmeans.softmax_plda_clust_adapt
adapt_cluster_start=$adapt_purify_cluster/20
adapt_cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.ahc.softmax_plda_clust_adapt
adapt_result=test/result_dvector_softmax_plda_clust_adapt

#adapt_purify_cluster=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.kmeans.softmax_plda.adapt
#adapt_cluster_start=$adapt_purify_cluster/20
#adapt_cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.ahc.softmax_plda.adapt
#adapt_result=test/result_dvector_softmax_plda_adapt

# Threshold used in episode clustering and entire dataset clustering
# Threshold when clustering within each show while using clustering results in the episodes.
init_intra_thresh=-4.5
init_inter_thresh=15
init_show_thresh=30

#scoring=cos
#
## Before adaptation
#nnetdir=$exp/xvector_nnet_tdnn_asoftmax_fn_s20_linear
#checkpoint=-1
#sub_path=decode/clust/cpdaic_1.0_50/lib/flists.nomerge
#purify_cluster=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.kmeans.asoftmax_cos
#cluster_start=$purify_cluster/20
#cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.ahc.asoftmax_cos
#result=test/result_dvector_asoftmax_cos
#
## After adaptation
#data_adapt=$data/asoftmax_cos
#adapt_checkpoint=-1
#adapt_purify_cluster=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.kmeans.asoftmax_cos.adapt
#adapt_cluster_start=$adapt_purify_cluster/20
#adapt_cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.ahc.asoftmax_cos.adapt
#adapt_result=test/result_dvector_asoftmax_cos_adapt
#
## Threshold used in episode clustering and entire dataset clustering
## Threshold when clustering within each show while using clustering results in the episodes.
#init_intra_thresh=115
#init_inter_thresh=350
#init_show_thresh=350



if [ $stage -le 29 ]; then
  # Test the linked performance
  # Initial clustering within each episode. Dev15 and dev15l can be used to check the clustering status.
  init_intra_thresh=-7.0
  init_show_thresh=25
  linked_start=$adapt_cluster_start
  scoring=plda_adapt

#  cd $base
#  base-diar/run/step-cpdaic-kmedoids $task plda_clust_adapt 20 $nnetdir/xvectors_$task $sub_path $adapt_purify_cluster
#  cd -
#
#  cd $base
#  base-diar/run/step-ahc-mat -THRESH $init_intra_thresh $task $scoring $nnetdir/xvectors_$task $linked_start $cluster_result/final
#  base-diar/run/step-eval-cluster $task $cluster_result/final $result/final notlinked
#  cd -

#  mkdir -p $nnetdir/xvectors_${task}/adapt/final
#  sid/ivector_nnet/create_semisupervised_data.sh $task $cluster_result/final $data/${task} $data_adapt/${task}_final
#  utils/apply_map.pl -f 1 $data_adapt/${task}_final/utt_map < $nnetdir/xvectors_${task}/xvector.scp > $nnetdir/xvectors_${task}/adapt/final/xvector.scp
#
#  # Clustering within shows using the episode clustering results.
#  for show in `cut -d '-' -f 1 ${base}/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
#    rm -rf $nnetdir/xvectors_${task}/adapt/final/$show $data_adapt/${task}_final/$show
#    mkdir -p $data_adapt/${task}_final/$show $nnetdir/xvectors_${task}/adapt/final/$show
#    grep `echo $show` $data_adapt/${task}_final/spk2utt > $data_adapt/${task}_final/$show/spk2utt
#    grep `echo $show` $nnetdir/xvectors_${task}/adapt/final/xvector.scp > $nnetdir/xvectors_${task}/adapt/final/$show/xvector.scp
#
#    # adapt_lambda is only used when plda_adapt is applied as the scoring method.
#    sid/ivector_nnet/speaker_pairwise_score.sh --cmd "$train_cmd" --adapt-lambda "$nnetdir/xvectors_dev15"\
#      $scoring $nnetdir/xvectors_voxceleb_train $data_adapt/${task}_final/$show $nnetdir/xvectors_${task}/adapt/final/$show &
#  done
#  wait

  for show in `cut -d '-' -f 1 ${base}/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
    $train_cmd $nnetdir/xvectors_${task}/adapt/final/$show/log/spk_clustering.log \
      python base-diar/python/spk_ahc_cluster_mat.py -p $show -t $init_show_thresh $nnetdir/xvectors_${task}/adapt/final/$show/score_${scoring}.txt $nnetdir/xvectors_${task}/adapt/final/$show/spk_ahc_${scoring} &
  done
  wait

  rm -f $nnetdir/xvectors_${task}/adapt/final/show_spk_ahc_${scoring}
  for show in `cut -d '-' -f 1 ${base}/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
    cat $nnetdir/xvectors_${task}/adapt/final/$show/spk_ahc_${scoring}
  done >> $nnetdir/xvectors_${task}/adapt/final/show_spk_ahc_${scoring}

  sid/ivector_nnet/speaker_cluster_data.sh $nnetdir/xvectors_${task}/adapt/final/show_spk_ahc_${scoring} $data_adapt/${task}_final $data_adapt/${task}_final_show_ahc

  cd $base
  base-diar/run/step-eval-spk-cluster $task $data_adapt/${task}_final_show_ahc/utt2spk $result/final_show_ahc notlinked
  base-diar/run/step-eval-spk-cluster $task $data_adapt/${task}_final_show_ahc/utt2spk $result/final_show_ahc linked
  cd -
  exit 1
fi


if [ $stage -le 30 ]; then
  # Test the baseline Kaldi softmax
  purify_cluster=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.kmeans.kaldi
  cluster_start=$purify_cluster/20
  cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.ahc.kaldi
  result=test/result_dvector_kaldi
  kaldi_nnet_dir=exp/xvector_nnet_1a

#  sid/nnet3/xvector/extract_xvectors_mgb.sh --cmd "$train_cmd" --nj 40 \
#    $kaldi_nnet_dir $data/$task $exp/xvectors_$task

#  for show in `cat $base/lib/flists.test/$task.lst`; do
#    mkdir -p $exp/xvectors_$task/$show
#    grep `echo $show | cut -d '_' -f 2` $exp/xvectors_$task/xvector.scp > $exp/xvectors_$task/$show/xvector.scp
#    if [ -s $exp/xvectors_$task/$show/xvector.scp ]; then
#      python base-diar/python/create_pairwise_trials.py $exp/xvectors_$task/$show/xvector.scp $exp/xvectors_$task/$show/trials &
#      awk -F ' ' '{print $1" 1"}' $exp/xvectors_$task/$show/xvector.scp > $exp/xvectors_$task/$show/num_utts.ark
#    else
#      echo "File $exp/xvectors_$task/$show/xvector.scp is empty"
#    fi
#  done
#  wait

#  for show in `cat $base/lib/flists.test/$task.lst`; do
#    if [ -s $exp/xvectors_$task/$show/trials ]; then
#      $train_cmd $exp/xvectors_$task/$show/log/plda_scoring.log \
#      ivector-plda-scoring --normalize-length=true \
#        --num-utts=ark:$exp/xvectors_$task/$show/num_utts.ark \
#        "ivector-copy-plda --smoothing=0.0 $kaldi_nnet_dir/xvectors_train/plda - |" \
#        "ark:ivector-subtract-global-mean $kaldi_nnet_dir/xvectors_train/mean.vec scp:$exp/xvectors_$task/$show/xvector.scp ark:- | transform-vec $kaldi_nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#        "ark:ivector-subtract-global-mean $kaldi_nnet_dir/xvectors_train/mean.vec scp:$exp/xvectors_$task/$show/xvector.scp ark:- | transform-vec $kaldi_nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#        $exp/xvectors_$task/$show/trials $exp/xvectors_$task/$show/score_plda.txt &
#    fi
#  done
#  wait

#  cd $base
#  rm -rf $result
#  base-diar/run/step-cpdaic-kmedoids $task plda 20 $exp/xvectors_$task $sub_path $purify_cluster
#  base-diar/run/step-eval-kmedoids $task 10 $purify_cluster $result notlinked
#  cd -

#  cd $base
#  base-diar/run/step-score-eer $task plda $exp/xvectors_$task $sub_path
#  base-diar/run/step-score-cluster-eer $task plda normal normal $exp/xvectors_$task $sub_path
#  base-diar/run/step-score-cluster-eer $task plda center center $exp/xvectors_$task $sub_path
#  cd -

#  cd $base
#  base-diar/run/ahc-full-thresh-mat -4.5 0.5 -1.0 $task plda $exp/xvectors_$task $sub_path $cluster_result $result notlinked
#  cd -

#  cd $base
#  base-diar/run/ahc-full-thresh-mat -3.0 0.5 -1.5 $task plda $exp/xvectors_$task $cluster_start $cluster_result $result notlinked
#  cd -

  cd $base
  base-diar/run/ahc-full-thresh-mat -ORG -3.0 0.5 -1.5 $task plda $exp/xvectors_$task $cluster_start $cluster_result $result notlinked
  cd -
  exit 1
fi