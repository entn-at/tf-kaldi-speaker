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

stage=29
task=dev15l

# The directory for dev15l, dev15 and train set.
base=/home/dawna/mgb3/transcription/exp-yl695/Snst/decode-$task
base_dev15=/home/dawna/mgb3/transcription/exp-yl695/Snst/decode-dev15
base_train=/home/dawna/mgb3/transcription/exp-yl695/Snst/decode-train

# If from_org=-ORG, the clustering is from scratch rather than from AIC results.
#from_org=-ORG

scoring=plda
aic_path=decode/clust/cpdaic_1.0_50/lib/flists.nomerge

nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2_m0.9
checkpoint=-1
purify_cluster=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.kmeans.softmax_plda
cluster_start=$purify_cluster/20
cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.ahc.softmax_plda
result=test/result_dvector_softmax_plda

adapt_nnetdir=$nnetdir/adapt/stage_two
data_adapt=$data/softmax_plda
adapt_checkpoint=5000
adapt_purify_cluster=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.kmeans.softmax_plda.adapt
adapt_cluster_start=$adapt_purify_cluster/20
adapt_cluster_result=decode/clust/cpdaic_1.0_50/lib/flists.nomerge.xvector.ahc.softmax_plda.adapt
adapt_result=test/result_dvector_softmax_plda_adapt

# Threshold used in episode clustering and entire dataset clustering
# Threshold when clustering within each show while using clustering results in the episodes.
init_intra_thresh=-4.5
init_inter_thresh=15
init_show_thresh=30


if [ $stage -le -1 ]; then
    # Link the directories from kaldi egs/voxceleb
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
  # Make kaldi data directory from the CPD-AIC result.
  # Get wav.scp, segments, utt2spk and spk2utt
  mkdir -p $data/$task
  awk -F '[/-]' '{print $(NF-1)}' $feat_dir/lib/coding/${task}.fbk | sed 's/_/-/g' | paste -d ' ' $feat_dir/lib/coding/${task}.fbk - | awk '{print $3" "$1}' > $data/$task/wav.scp
  rm -f $data/$task/segments $data/$task/utt2spk
  for show in `cat $base/lib/flists.test/$task.lst`; do
    awk -F '[.=_-]' '{print $1"-"$3"-"$4"-"$7"-"$8}' $base/test/$task/${show}.1/$aic_path/${show}.scp | awk '{print $1" "$1}' >> $data/$task/utt2spk
    cp $data/$task/utt2spk $data/$task/spk2utt
    awk -F '-' '{print $0" "$1"-"$2"-"$3" "$(NF-1)/100" "$(NF)/100}' $data/$task/utt2spk |  awk '{print $1" "$3" "$4" "$5}' >> $data/$task/segments
  done
  # Do I need to modify the segments to match the actual duration?
  utils/fix_segments.sh $data/$task
  utils/fix_data_dir.sh $data/$task
fi

if [ $stage -le 1 ]; then
  # Extract features for dev15l (the test set).
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $data/$task $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data/$task
fi


if [ $stage -le 2 ]; then
  # Extract the embeddings (d-vector).
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/$task $nnetdir/xvectors_$task
fi


if [ $stage -le 3 ]; then
  # Scoring using PLDA
  # If the embedding is extracted from tdnn6 (the second last layer), PLDA must be applied.
  # If the embedding is extracted from tdnn7 (the last layer), cosine is an okay choice.
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

  if [ $scoring == 'plda' ]; then
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
  elif [ $scoring == 'cos' ]; then
    # Scoring using cosine similarity
    for show in `cat $base/lib/flists.test/$task.lst`; do
      if [ -s $nnetdir/xvectors_${task}/$show/trials ]; then
        $train_cmd $nnetdir/xvectors_${task}/$show/log/cos_scoring.log \
            ivector-compute-dot-products "cat $nnetdir/xvectors_${task}/$show/trials | cut -d\  --fields=1,2 |"  \
            "ark:ivector-normalize-length scp:$nnetdir/xvectors_${task}/$show/xvector.scp ark:- |" \
            "ark:ivector-normalize-length scp:$nnetdir/xvectors_${task}/$show/xvector.scp ark:- |" \
            $nnetdir/xvectors_${task}/$show/score_cos.txt &
      fi
    done
    wait
  fi
fi

if [ $stage -le 4 ]; then
  # Cluster purification using k-medoids
  # The performance can be evaluated on dev15l
  cd $base
  rm -rf $result
  base-diar/run/step-cpdaic-kmedoids $task $scoring 20 $nnetdir/xvectors_$task $aic_path $purify_cluster
  base-diar/run/step-eval-kmedoids $task 10 $purify_cluster $result notlinked
  cd -
fi


if [ $stage -le 5 ]; then
  # Evaluate the pairwise EER. It seems that the `center-center` result is a good metric and has a relation with the DER performance.
  cd $base
  base-diar/run/step-score-eer $task $scoring $nnetdir/xvectors_$task $cluster_start
  base-diar/run/step-score-cluster-eer $task $scoring normal normal $nnetdir/xvectors_$task $cluster_start
  base-diar/run/step-score-cluster-eer $task $scoring center center $nnetdir/xvectors_$task $cluster_start
  cd -
fi

if [ $stage -le 6 ]; then
  # Do clustering. The range of the threshold should be set manually.
  # Set from_org=-ORG to do clustering from scratch.
  cd $base
  base-diar/run/ahc-full-thresh-mat $from_org -5.0 0.5 -2.0 $task $scoring $nnetdir/xvectors_$task $cluster_start $cluster_result $result notlinked
  cd -
fi


# PLDA adaptation
if [ $stage -le 7 ]; then
  # Use dev15 to adapt the PLDA to see what's going on.
  mkdir -p $data/dev15
  awk -F '[/-]' '{print $(NF-1)}' $feat_dir/lib/coding/dev15.fbk | sed 's/_/-/g' | paste -d ' ' $feat_dir/lib/coding/dev15.fbk - | awk '{print $3" "$1}' > $data/dev15/wav.scp
  rm -f $data/dev15/segments $data/dev15/utt2spk
  for show in `cat $base_dev15/lib/flists.test/dev15.lst`; do
    awk -F '[.=_-]' '{print $1"-"$3"-"$4"-"$7"-"$8}' $base_dev15/test/dev15/${show}.1/$aic_path/${show}.scp | awk '{print $1" "$1}' >> $data/dev15/utt2spk
    cp $data/dev15/utt2spk $data/dev15/spk2utt
    awk -F '-' '{print $0" "$1"-"$2"-"$3" "$(NF-1)/100" "$(NF)/100}' $data/dev15/utt2spk |  awk '{print $1" "$3" "$4" "$5}' >> $data/dev15/segments
  done
  # Do I need to modify the segments to match the actual duration?
  utils/fix_segments.sh $data/dev15
  utils/fix_data_dir.sh $data/dev15
fi

if [ $stage -le 8 ]; then
  # Extract features on dev15
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $data/dev15 $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data/dev15
fi

if [ $stage -le 9 ]; then
  # Extract the embeddings (d-vector)
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/dev15 $nnetdir/xvectors_dev15
fi

if [ $stage -le 10 ]; then
  # Kaldi un-supervised adaptation
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
  # Scoring using adapted PLDA
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


if [ $stage -le 12 ]; then
  # Test the pairwise EER.
  cd $base
  base-diar/run/step-score-eer $task plda_adapt $nnetdir/xvectors_$task $cluster_start
  base-diar/run/step-score-cluster-eer $task plda_adapt normal normal $nnetdir/xvectors_$task $cluster_start
  base-diar/run/step-score-cluster-eer $task plda_adapt center center $nnetdir/xvectors_$task $cluster_start
  cd -

  cd $base
  base-diar/run/ahc-full-thresh-mat $from_org -9.0 0.5 -6.0 $task plda_adapt $nnetdir/xvectors_$task $cluster_start $cluster_result $result notlinked
  cd -
fi


# Neural network adaptation
if [ $stage -le 13 ]; then
  # Train set should be used in the neural adaptation.
  # Extract features for train set
  mkdir -p $data/train
  awk -F '[/-]' '{print $(NF-1)}' $feat_dir/lib/coding/train.fbk | sed 's/_/-/g' | paste -d ' ' $feat_dir/lib/coding/train.fbk - | awk '{print $3" "$1}' > $data/train/wav.scp
  rm -f $data/train/segments $data/train/utt2spk
  for show in `cat $base_train/lib/flists.test/train.lst`; do
    awk -F '[.=_-]' '{print $1"-"$3"-"$4"-"$7"-"$8}' $base_train/test/train/${show}.1/$aic_path/${show}.scp | awk '{print $1" "$1}' >> $data/train/utt2spk
    cp $data/train/utt2spk $data/train/spk2utt
    awk -F '-' '{print $0" "$1"-"$2"-"$3" "$(NF-1)/100" "$(NF)/100}' $data/train/utt2spk |  awk '{print $1" "$3" "$4" "$5}' >> $data/train/segments
  done
  # Do I need to modify the segments to match the actual duration?
  utils/fix_segments.sh $data/train
  utils/fix_data_dir.sh $data/train
fi

if [ $stage -le 14 ]; then
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $data/train $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data/train
fi

if [ $stage -le 15 ]; then
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 100 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/train $nnetdir/xvectors_train
fi

if [ $stage -le 16 ]; then
  # Prepare the trials list to the following scoring
  for name in dev15 train; do
    dir=base_${name}
    for show in `cat ${!dir}/lib/flists.test/${name}.lst`; do
      mkdir -p $nnetdir/xvectors_${name}/$show
      grep `echo $show | cut -d '_' -f 2` $nnetdir/xvectors_${name}/xvector.scp > $nnetdir/xvectors_${name}/$show/xvector.scp
      if [ -s $nnetdir/xvectors_${name}/$show/xvector.scp ]; then
        python base-diar/python/create_pairwise_trials.py $nnetdir/xvectors_${name}/$show/xvector.scp $nnetdir/xvectors_${name}/$show/trials &
        awk -F ' ' '{print $1" 1"}' $nnetdir/xvectors_${name}/$show/xvector.scp > $nnetdir/xvectors_${name}/$show/num_utts.ark
      else
        echo "File $nnetdir/xvectors_${name}/$show/xvector.scp is empty"
      fi
    done
    wait
  done
fi

if [ $stage -le 17 ]; then
  # Compute PLDA on dev15 and train set. We use dev15 as the dev set.
  if [ $scoring == 'plda' ]; then
    for name in dev15 train; do
      dir=base_${name}
      for show in `cat ${!dir}/lib/flists.test/${name}.lst`; do
        if [ -s $nnetdir/xvectors_${name}/$show/trials ]; then
          $train_cmd $nnetdir/xvectors_${name}/$show/log/plda_scoring.log \
            ivector-plda-scoring --normalize-length=true \
            --num-utts=ark:$nnetdir/xvectors_${name}/$show/num_utts.ark \
            "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_voxceleb_train/plda - |" \
            "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_${name}/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
            "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_${name}/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
            $nnetdir/xvectors_${name}/$show/trials $nnetdir/xvectors_${name}/$show/score_plda.txt &
        fi
      done
      wait
    done
  elif [ $scoring == 'cos' ]; then
    # Also Cosine similarity.
    for name in dev15 train; do
      dir=base_${name}
      for show in `cat ${!dir}/lib/flists.test/${name}.lst`; do
        if [ -s $nnetdir/xvectors_${name}/$show/trials ]; then
          $train_cmd $nnetdir/xvectors_${name}/$show/log/cos_scoring.log \
              ivector-compute-dot-products "cat $nnetdir/xvectors_${name}/$show/trials | cut -d\  --fields=1,2 |"  \
              "ark:ivector-normalize-length scp:$nnetdir/xvectors_${name}/$show/xvector.scp ark:- |" \
              "ark:ivector-normalize-length scp:$nnetdir/xvectors_${name}/$show/xvector.scp ark:- |" \
              $nnetdir/xvectors_${name}/$show/score_cos.txt &
        fi
      done
      wait
    done
  fi
fi

if [ $stage -le 18 ]; then
  # Purify the AIC result on dev15 and train set.
  cd $base_dev15
  rm -rf $result
  base-diar/run/step-cpdaic-kmedoids dev15 $scoring 20 $nnetdir/xvectors_dev15 $aic_path $purify_cluster
  base-diar/run/step-eval-kmedoids dev15 10 $purify_cluster $result notlinked
  cd -
  cd $base_train
  base-diar/run/step-cpdaic-kmedoids train $scoring 20 $nnetdir/xvectors_train $aic_path $purify_cluster
  cd -
fi


if [ $stage -le 19 ]; then
  # Initial clustering within each episode. Dev15l can be used to check the clustering status.
  cd $base
  base-diar/run/step-ahc-mat $from_org -THRESH $init_intra_thresh $task $scoring $nnetdir/xvectors_$task $cluster_start $cluster_result/0
  base-diar/run/step-eval-cluster $task $cluster_result/0 $result/0 notlinked
  base-diar/run/step-eval-cluster $task $cluster_result/0 $result/0 linked
  cd -
fi

if [ $stage -le 20 ]; then
  # Test the linkage performance on dev15l
  # Clustering across the entire dataset.
  rm -rf $nnetdir/xvectors_${task}/adapt/0.bak $nnetdir/xvectors_${task}/adapt/0_ahc.bak $data_adapt/${task}_0.bak $data_adapt/${task}_0_ahc.bak
  mv $data_adapt/${task}_0 $data_adapt/${task}_0.bak
  mv $data_adapt/${task}_0_ahc $data_adapt/${task}_0_ahc.bak
  mv $nnetdir/xvectors_${task}/adapt/0 $nnetdir/xvectors_${task}/adapt/0.bak
  mv $nnetdir/xvectors_${task}/adapt/0_ahc $nnetdir/xvectors_${task}/adapt/0_ahc.bak

  mkdir -p $nnetdir/xvectors_${task}/adapt/0
  sid/ivector_nnet/create_semisupervised_data.sh $task $cluster_result/0 $data/${task} $data_adapt/${task}_0
  utils/apply_map.pl -f 1 $data_adapt/${task}_0/utt_map < $nnetdir/xvectors_${task}/xvector.scp > $nnetdir/xvectors_${task}/adapt/0/xvector.scp
  sid/ivector_nnet/speaker_pairwise_score.sh --cmd "$train_cmd" \
    $scoring $nnetdir/xvectors_voxceleb_train $data_adapt/${task}_0 $nnetdir/xvectors_${task}/adapt/0
  python base-diar/python/spk_ahc_cluster_mat.py -t $init_inter_thresh $nnetdir/xvectors_${task}/adapt/0/score_${scoring}.txt $nnetdir/xvectors_${task}/adapt/0/spk_ahc_${scoring}
  sid/ivector_nnet/speaker_cluster_data.sh $nnetdir/xvectors_${task}/adapt/0/spk_ahc_${scoring} $data_adapt/${task}_0 $data_adapt/${task}_0_ahc

  cd $base
  base-diar/run/step-eval-spk-cluster $task $data_adapt/${task}_0_ahc/utt2spk $result/0_ahc notlinked
  base-diar/run/step-eval-spk-cluster $task $data_adapt/${task}_0_ahc/utt2spk $result/0_ahc linked
  cd -
fi


if [ $stage -le 21 ]; then
  # Clustering within shows using the episode clustering results.
  for show in `cut -d '-' -f 1 $base/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
    rm -rf $nnetdir/xvectors_${task}/adapt/0/$show $data_adapt/${task}_0/$show

    # Process the data and xvector directories
    mkdir -p $data_adapt/${task}_0/$show $nnetdir/xvectors_${task}/adapt/0/$show
    grep `echo $show` $data_adapt/${task}_0/spk2utt > $data_adapt/${task}_0/$show/spk2utt
    grep `echo $show` $nnetdir/xvectors_${task}/adapt/0/xvector.scp > $nnetdir/xvectors_${task}/adapt/0/$show/xvector.scp
    sid/ivector_nnet/speaker_pairwise_score.sh --cmd "$train_cmd" \
      $scoring $nnetdir/xvectors_voxceleb_train $data_adapt/${task}_0/$show $nnetdir/xvectors_${task}/adapt/0/$show
    python base-diar/python/spk_ahc_cluster_mat.py -p $show -t $init_show_thresh $nnetdir/xvectors_${task}/adapt/0/$show/score_${scoring}.txt $nnetdir/xvectors_${task}/adapt/0/$show/spk_ahc_${scoring}
  done
  rm -f $nnetdir/xvectors_${task}/adapt/0/show_spk_ahc_${scoring}
  for show in `cut -d '-' -f 1 $base/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
    cat $nnetdir/xvectors_${task}/adapt/0/$show/spk_ahc_${scoring}
  done >> $nnetdir/xvectors_${task}/adapt/0/show_spk_ahc_${scoring}
  sid/ivector_nnet/speaker_cluster_data.sh $nnetdir/xvectors_${task}/adapt/0/show_spk_ahc_${scoring} $data_adapt/${task}_0 $data_adapt/${task}_0_show_ahc

  cd $base
  base-diar/run/step-eval-spk-cluster $task $data_adapt/${task}_0_show_ahc/utt2spk $result/0_show_ahc notlinked
  base-diar/run/step-eval-spk-cluster $task $data_adapt/${task}_0_show_ahc/utt2spk $result/0_show_ahc linked
  cd -
fi

#if [ $stage -le 22 ]; then
#  # Clustering within shows and across episodes.
#  # Generate a new score file before clustering within shows.
#  # It seems that clustering on the show performs badly. Clustering on the episode and then clustering on the show is a better choice.
#  for show in `cut -d '-' -f 1 $base/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
#    mkdir -p $nnetdir/xvectors_$task/$show
#    grep $show $nnetdir/xvectors_$task/xvector.scp > $nnetdir/xvectors_$task/$show/xvector.scp
#    if [ -s $nnetdir/xvectors_$task/$show/xvector.scp ]; then
#      python base-diar/python/create_pairwise_trials.py $nnetdir/xvectors_$task/$show/xvector.scp $nnetdir/xvectors_$task/$show/trials &
#      awk -F ' ' '{print $1" 1"}' $nnetdir/xvectors_$task/$show/xvector.scp > $nnetdir/xvectors_$task/$show/num_utts.ark
#    else
#      echo "File $nnetdir/xvectors_$task/$show/xvector.scp is empty"
#    fi
#  done
#  wait
#
#  for show in `cut -d '-' -f 1 $base/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
#    if [ -s $nnetdir/xvectors_$task/$show/trials ]; then
#      $train_cmd $nnetdir/xvectors_$task/$show/log/plda_scoring.log \
#        ivector-plda-scoring --normalize-length=true \
#        --num-utts=ark:$nnetdir/xvectors_$task/$show/num_utts.ark \
#        "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_voxceleb_train/plda - |" \
#        "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#        "ark:ivector-subtract-global-mean $nnetdir/xvectors_voxceleb_train/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#        $nnetdir/xvectors_$task/$show/trials $nnetdir/xvectors_$task/$show/score_plda.txt &
#    fi
#  done
#  wait
#
#  # Scoring using cosine similarity
#  for show in `cut -d '-' -f 1 $base/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
#    if [ -s $nnetdir/xvectors_${task}/$show/trials ]; then
#      $train_cmd $nnetdir/xvectors_${task}/$show/log/cos_scoring.log \
#          ivector-compute-dot-products "cat $nnetdir/xvectors_${task}/$show/trials | cut -d\  --fields=1,2 |"  \
#          "ark:ivector-normalize-length scp:$nnetdir/xvectors_${task}/$show/xvector.scp ark:- |" \
#          "ark:ivector-normalize-length scp:$nnetdir/xvectors_${task}/$show/xvector.scp ark:- |" \
#          $nnetdir/xvectors_${task}/$show/score_cos.txt &
#    fi
#  done
#  wait
#
#  cd $base
#  base-diar/run/show-ahc-full-thresh-mat $from_org -10.0 2.5 -5.0 $task $scoring $nnetdir/xvectors_$task $cluster_start $cluster_result $result notlinked
#  cd -
#
#  cd $base
#  base-diar/run/step-show-ahc-mat $from_org -THRESH $init_intra_thresh $task $scoring $nnetdir/xvectors_$task $cluster_start $cluster_result/show_0
#  cd -
#
#  cd $base
#  base-diar/run/step-eval-cluster $task $cluster_result/show_0 $result/show_0 notlinked
#  base-diar/run/step-eval-cluster $task $cluster_result/show_0 $result/show_0 linked
#  cd -
#  exit 1
#fi


if [ $stage -le 23 ]; then
  # Linking on dev15 and train set.
  cd $base_train
  base-diar/run/step-ahc-mat $from_org -THRESH $init_intra_thresh train $scoring $nnetdir/xvectors_train $cluster_start $cluster_result/0
  cd -
  cd $base_dev15
  base-diar/run/step-ahc-mat $from_org -THRESH $init_intra_thresh dev15 $scoring $nnetdir/xvectors_dev15 $cluster_start $cluster_result/0
  base-diar/run/step-eval-cluster dev15 $cluster_result/0 $result/0
  cd -
fi


if [ $stage -le 24 ]; then
  # Train set is too large to cluster. So only within show clustering is applied.
  for name in train dev15; do
    dir=base_${name}
    rm -rf $nnetdir/xvectors_${name}/adapt/0.bak $nnetdir/xvectors_${name}/adapt/0_ahc.bak $data_adapt/${name}_0.bak $data_adapt/${name}_0_ahc.bak
    mv $data_adapt/${name}_0 $data_adapt/${name}_0.bak
    mv $data_adapt/${name}_0_ahc $data_adapt/${name}_0_ahc.bak
    mv $nnetdir/xvectors_${name}/adapt/0 $nnetdir/xvectors_${name}/adapt/0.bak
    mv $nnetdir/xvectors_${name}/adapt/0_ahc $nnetdir/xvectors_${name}/adapt/0_ahc.bak

    mkdir -p $nnetdir/xvectors_${name}/adapt/0
    sid/ivector_nnet/create_semisupervised_data.sh $name $cluster_result/0 $data/${name} $data_adapt/${name}_0
    utils/apply_map.pl -f 1 $data_adapt/${name}_0/utt_map < $nnetdir/xvectors_${name}/xvector.scp > $nnetdir/xvectors_${name}/adapt/0/xvector.scp

    for show in `cut -d '-' -f 1 ${!dir}/lib/flists.test/${name}.lst | cut -d '_' -f 2 | sort -u`; do
      rm -rf $nnetdir/xvectors_${name}/adapt/0/$show $data_adapt/${name}_0/$show

      # Process the data and xvector directories
      mkdir -p $data_adapt/${name}_0/$show $nnetdir/xvectors_${name}/adapt/0/$show
      grep `echo $show` $data_adapt/${name}_0/spk2utt > $data_adapt/${name}_0/$show/spk2utt
      grep `echo $show` $nnetdir/xvectors_${name}/adapt/0/xvector.scp > $nnetdir/xvectors_${name}/adapt/0/$show/xvector.scp

      sid/ivector_nnet/speaker_pairwise_score.sh --cmd "$train_cmd" \
        $scoring $nnetdir/xvectors_voxceleb_train $data_adapt/${name}_0/$show $nnetdir/xvectors_${name}/adapt/0/$show &
    done
    wait

    for show in `cut -d '-' -f 1 ${!dir}/lib/flists.test/${name}.lst | cut -d '_' -f 2 | sort -u`; do
      $train_cmd $nnetdir/xvectors_${name}/adapt/0/$show/log/spk_clustering.log \
        python base-diar/python/spk_ahc_cluster_mat.py -p $show -t $init_show_thresh $nnetdir/xvectors_${name}/adapt/0/$show/score_${scoring}.txt $nnetdir/xvectors_${name}/adapt/0/$show/spk_ahc_${scoring} &
    done
    wait

    rm -f $nnetdir/xvectors_${name}/adapt/0/show_spk_ahc_${scoring}
    for show in `cut -d '-' -f 1 ${!dir}/lib/flists.test/${name}.lst | cut -d '_' -f 2 | sort -u`; do
      cat $nnetdir/xvectors_${name}/adapt/0/$show/spk_ahc_${scoring}
    done >> $nnetdir/xvectors_${name}/adapt/0/show_spk_ahc_${scoring}

    sid/ivector_nnet/speaker_cluster_data.sh $nnetdir/xvectors_${name}/adapt/0/show_spk_ahc_${scoring} $data_adapt/${name}_0 $data_adapt/${name}_0_show_ahc
  done
fi


if [ $stage -le 25 ]; then
  # Prepare the features to fine-tune the network.
  # In order to train the network, cmvn should be applied to the features
  for name in dev15 dev15l train; do
    local/nnet3/xvector/prepare_feats_for_egs_mgb.sh --nj 40 --cmd "$train_cmd" \
      $data_adapt/${name}_0_show_ahc $data_adapt/${name}_0_show_ahc_no_sil $data_adapt/${name}_0_show_ahc_no_sil
  done

  sid/ivector_nnet/create_finetune_dev.sh dev15l $base $cluster_start $data/dev15l_dev
  local/nnet3/xvector/prepare_feats_for_egs_mgb.sh --nj 40 --cmd "$train_cmd" \
    $data/dev15l_dev $data/dev15l_dev_no_sil $exp/dev15l_dev_no_sil

  # If there is the number of utterances is too small, the speaker tends to be noisy.
  min_num_utts=5
  min_len=100
  for name in dev15 dev15l train; do
    feat-to-len scp:$data_adapt/${name}_0_show_ahc_no_sil/feats.scp ark,t:$data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames

    awk '{print $1, NF-1}' $data_adapt/${name}_0_show_ahc_no_sil/spk2utt > $data_adapt/${name}_0_show_ahc_no_sil/spk2num
    awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data_adapt/${name}_0_show_ahc_no_sil/spk2num | utils/filter_scp.pl - $data_adapt/${name}_0_show_ahc_no_sil/spk2utt > $data_adapt/${name}_0_show_ahc_no_sil/spk2utt.new
    mv $data_adapt/${name}_0_show_ahc_no_sil/spk2utt.new $data_adapt/${name}_0_show_ahc_no_sil/spk2utt
    utils/spk2utt_to_utt2spk.pl $data_adapt/${name}_0_show_ahc_no_sil/spk2utt > $data_adapt/${name}_0_show_ahc_no_sil/utt2spk
    utils/filter_scp.pl $data_adapt/${name}_0_show_ahc_no_sil/utt2spk $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames > $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames.new
    mv $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames.new $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames
    utils/fix_data_dir.sh $data_adapt/${name}_0_show_ahc_no_sil

    mv $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames.bak
    awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames.bak > $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames
    utils/filter_scp.pl $data_adapt/${name}_0_show_ahc_no_sil/utt2num_frames $data_adapt/${name}_0_show_ahc_no_sil/utt2spk > $data_adapt/${name}_0_show_ahc_no_sil/utt2spk.new
    mv $data_adapt/${name}_0_show_ahc_no_sil/utt2spk.new $data_adapt/${name}_0_show_ahc_no_sil/utt2spk
    utils/fix_data_dir.sh $data_adapt/${name}_0_show_ahc_no_sil
  done

  # And also for the dev set.
  feat-to-len scp:$data/dev15l_dev_no_sil/feats.scp ark,t:$data/dev15l_dev_no_sil/utt2num_frames
  mv $data/dev15l_dev_no_sil/utt2num_frames $data/dev15l_dev_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/dev15l_dev_no_sil/utt2num_frames.bak > $data/dev15l_dev_no_sil/utt2num_frames
  utils/filter_scp.pl $data/dev15l_dev_no_sil/utt2num_frames $data/dev15l_dev_no_sil/utt2spk > $data/dev15l_dev_no_sil/utt2spk.new
  mv $data/dev15l_dev_no_sil/utt2spk.new $data/dev15l_dev_no_sil/utt2spk
  utils/fix_data_dir.sh $data/dev15l_dev_no_sil
fi


if [ $stage -le 26 ]; then
  iter=0

  train_set=train
  awk -v id=0 '{print $1, id++}' $data_adapt/${train_set}_0_show_ahc_no_sil/spk2utt > $data_adapt/${train_set}_0_show_ahc_no_sil/spklist
  awk -v id=0 '{print $1, id++}' $data/dev15l_dev_no_sil/spk2utt > $data/dev15l_dev_no_sil/spklist

  python finetune_mgb.py --config nnet_conf/softmax_finetune.json \
    $data_adapt/${train_set}_0_show_ahc_no_sil $data_adapt/${train_set}_0_show_ahc_no_sil/spklist \
    $data/dev15l_dev_no_sil $data/dev15l_dev_no_sil/spklist \
    $nnetdir $nnetdir/adapt
  exit 1
fi

if [ $stage -le 27 ]; then
  # We need to re-train the PLDA!!!
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 150 --use-gpu false --checkpoint $adapt_checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $adapt_nnetdir $data/voxceleb_train $adapt_nnetdir/xvectors_voxceleb_train

  $train_cmd $adapt_nnetdir/xvectors_voxceleb_train/log/compute_mean.log \
    ivector-mean scp:$adapt_nnetdir/xvectors_voxceleb_train/xvector.scp \
    $adapt_nnetdir/xvectors_voxceleb_train/mean.vec || exit 1;

  lda_dim=200
  $train_cmd $adapt_nnetdir/xvectors_voxceleb_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$adapt_nnetdir/xvectors_voxceleb_train/xvector.scp ark:- |" \
    ark:$data/voxceleb_train/utt2spk $adapt_nnetdir/xvectors_voxceleb_train/transform.mat || exit 1;

  $train_cmd $adapt_nnetdir/xvectors_voxceleb_train/log/plda.log \
    ivector-compute-plda ark:$data/voxceleb_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$adapt_nnetdir/xvectors_voxceleb_train/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $adapt_nnetdir/xvectors_voxceleb_train/plda || exit 1;

  # And extract x-vectors on dev15l and train set.
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $adapt_checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $adapt_nnetdir $data/$task $adapt_nnetdir/xvectors_$task

  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 100 --use-gpu false --checkpoint $adapt_checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $adapt_nnetdir $data_adapt/train_0_show_ahc $adapt_nnetdir/xvectors_train_0_show_ahc

  # Scoring using PLDA on the new x-vectors.
  for show in `cat $base/lib/flists.test/$task.lst`; do
    mkdir -p $adapt_nnetdir/xvectors_$task/$show
    grep `echo $show | cut -d '_' -f 2` $adapt_nnetdir/xvectors_$task/xvector.scp > $adapt_nnetdir/xvectors_$task/$show/xvector.scp
    if [ -s $adapt_nnetdir/xvectors_$task/$show/xvector.scp ]; then
      python base-diar/python/create_pairwise_trials.py $adapt_nnetdir/xvectors_$task/$show/xvector.scp $adapt_nnetdir/xvectors_$task/$show/trials &
      awk -F ' ' '{print $1" 1"}' $adapt_nnetdir/xvectors_$task/$show/xvector.scp > $adapt_nnetdir/xvectors_$task/$show/num_utts.ark
    else
      echo "File $adapt_nnetdir/xvectors_$task/$show/xvector.scp is empty"
    fi
  done

  for show in `cat $base/lib/flists.test/$task.lst`; do
    if [ -s $adapt_nnetdir/xvectors_$task/$show/trials ]; then
      $train_cmd $adapt_nnetdir/xvectors_$task/$show/log/plda_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:$adapt_nnetdir/xvectors_$task/$show/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 $adapt_nnetdir/xvectors_voxceleb_train/plda - |" \
        "ark:ivector-subtract-global-mean $adapt_nnetdir/xvectors_voxceleb_train/mean.vec scp:$adapt_nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $adapt_nnetdir/xvectors_voxceleb_train/mean.vec scp:$adapt_nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $adapt_nnetdir/xvectors_$task/$show/trials $adapt_nnetdir/xvectors_$task/$show/score_plda.txt &
    fi
  done
  wait

  cd $base
  rm -rf $adapt_result
  base-diar/run/step-cpdaic-kmedoids $task plda 20 $adapt_nnetdir/xvectors_$task $aic_path $adapt_purify_cluster
  base-diar/run/step-eval-kmedoids $task 10 $adapt_purify_cluster $adapt_result notlinked
  cd -

  cd $base
  base-diar/run/step-score-eer $task plda $adapt_nnetdir/xvectors_$task $aic_path
  base-diar/run/step-score-cluster-eer $task plda normal normal $adapt_nnetdir/xvectors_$task $aic_path
  base-diar/run/step-score-cluster-eer $task plda center center $adapt_nnetdir/xvectors_$task $aic_path
  cd -

  cd $base
  base-diar/run/ahc-full-thresh-mat -4.5 0.5 -0.5 $task plda $adapt_nnetdir/xvectors_$task $adapt_cluster_start $adapt_cluster_result $adapt_result notlinked
  cd -

  # Re-train PLDA on the train set.
  min_num_utts=5
  cp -r $data_adapt/train_0_show_ahc $data_adapt/train_0_show_ahc.bak
  awk '{print $1, NF-1}' $data_adapt/train_0_show_ahc/spk2utt > $data_adapt/train_0_show_ahc/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data_adapt/train_0_show_ahc/spk2num | utils/filter_scp.pl - $data_adapt/train_0_show_ahc/spk2utt > $data_adapt/train_0_show_ahc/spk2utt.new
  mv $data_adapt/train_0_show_ahc/spk2utt.new $data_adapt/train_0_show_ahc/spk2utt
  utils/spk2utt_to_utt2spk.pl $data_adapt/train_0_show_ahc/spk2utt > $data_adapt/train_0_show_ahc/utt2spk
  utils/fix_data_dir.sh $data_adapt/train_0_show_ahc

  cp $adapt_nnetdir/xvectors_train_0_show_ahc/xvector.scp $adapt_nnetdir/xvectors_train_0_show_ahc/xvector.scp.bak
  utils/filter_scp.pl $data_adapt/train_0_show_ahc/utt2spk < $adapt_nnetdir/xvectors_train_0_show_ahc/xvector.scp.bak > $adapt_nnetdir/xvectors_train_0_show_ahc/xvector.scp

  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $adapt_nnetdir/xvectors_train_0_show_ahc/log/compute_mean.log \
    ivector-mean scp:$adapt_nnetdir/xvectors_train_0_show_ahc/xvector.scp \
    $adapt_nnetdir/xvectors_train_0_show_ahc/mean.vec || exit 1;

  lda_dim=200
  $train_cmd $adapt_nnetdir/xvectors_train_0_show_ahc/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$adapt_nnetdir/xvectors_train_0_show_ahc/xvector.scp ark:- |" \
    ark:$data_adapt/train_0_show_ahc/utt2spk $adapt_nnetdir/xvectors_train_0_show_ahc/transform.mat || exit 1;

  $train_cmd $adapt_nnetdir/xvectors_train_0_show_ahc/log/plda.log \
    ivector-compute-plda ark:$data_adapt/train_0_show_ahc/spk2utt \
    "ark:ivector-subtract-global-mean scp:$adapt_nnetdir/xvectors_train_0_show_ahc/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_train_0_show_ahc/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $adapt_nnetdir/xvectors_train_0_show_ahc/plda || exit 1;

  # Scoring using the new PLDA.
  for show in `cat $base/lib/flists.test/$task.lst`; do
    if [ -s $adapt_nnetdir/xvectors_$task/$show/trials ]; then
      $train_cmd $adapt_nnetdir/xvectors_$task/$show/log/plda_scoring_nnet_adapt.log \
        ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:$adapt_nnetdir/xvectors_$task/$show/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 $adapt_nnetdir/xvectors_train_0_show_ahc/plda - |" \
        "ark:ivector-subtract-global-mean $adapt_nnetdir/xvectors_train_0_show_ahc/mean.vec scp:$adapt_nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_train_0_show_ahc/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $adapt_nnetdir/xvectors_train_0_show_ahc/mean.vec scp:$adapt_nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_train_0_show_ahc/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $adapt_nnetdir/xvectors_$task/$show/trials $adapt_nnetdir/xvectors_$task/$show/score_plda_nnet_adapt.txt &
    fi
  done
  wait

  cd $base
  rm -rf $adapt_result
  base-diar/run/step-cpdaic-kmedoids $task plda_nnet_adapt 20 $adapt_nnetdir/xvectors_$task $aic_path $adapt_purify_cluster
  base-diar/run/step-eval-kmedoids $task 10 $adapt_purify_cluster $adapt_result notlinked
  cd -

  cd $base
  base-diar/run/step-score-eer $task plda_nnet_adapt $adapt_nnetdir/xvectors_$task $aic_path
  base-diar/run/step-score-cluster-eer $task plda_nnet_adapt normal normal $adapt_nnetdir/xvectors_$task $aic_path
  base-diar/run/step-score-cluster-eer $task plda_nnet_adapt center center $adapt_nnetdir/xvectors_$task $aic_path
  cd -

  # Test the performance.
  cd $base
  base-diar/run/ahc-full-thresh-mat -4.5 0.5 -0.5 $task plda $adapt_nnetdir/xvectors_$task $adapt_cluster_start $adapt_cluster_result $adapt_result notlinked
  cd -

  cd $base
  base-diar/run/ahc-full-thresh-mat -2.0 0.5 3.0 $task plda_nnet_adapt $adapt_nnetdir/xvectors_$task $adapt_cluster_start $adapt_cluster_result $adapt_result notlinked
  cd -

  # Do PLDA adaptation on the train set (rather than re-train PLDA).
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $adapt_checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $adapt_nnetdir $data/dev15 $adapt_nnetdir/xvectors_dev15

  $train_cmd $adapt_nnetdir/xvectors_dev15/log/compute_mean.log \
    ivector-mean scp:$adapt_nnetdir/xvectors_dev15/xvector.scp \
      $adapt_nnetdir/xvectors_dev15/mean.vec || exit 1;

  $train_cmd $adapt_nnetdir/xvectors_dev15/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $adapt_nnetdir/xvectors_voxceleb_train/plda \
    "ark:ivector-subtract-global-mean scp:$adapt_nnetdir/xvectors_dev15/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $adapt_nnetdir/xvectors_dev15/plda_adapt || exit 1;

  for show in `cat $base/lib/flists.test/$task.lst`; do
    if [ -s $adapt_nnetdir/xvectors_$task/$show/trials ]; then
      $train_cmd $adapt_nnetdir/xvectors_$task/$show/log/plda_adapt_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:$adapt_nnetdir/xvectors_$task/$show/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 $adapt_nnetdir/xvectors_dev15/plda_adapt - |" \
        "ark:ivector-subtract-global-mean $adapt_nnetdir/xvectors_dev15/mean.vec scp:$adapt_nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $adapt_nnetdir/xvectors_dev15/mean.vec scp:$adapt_nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $adapt_nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $adapt_nnetdir/xvectors_$task/$show/trials $adapt_nnetdir/xvectors_$task/$show/score_plda_adapt.txt &
    fi
  done
  wait

  cd $base
  base-diar/run/ahc-full-thresh-mat -9.0 0.5 -5.0 $task plda_adapt $adapt_nnetdir/xvectors_$task $adapt_cluster_start $adapt_cluster_result $adapt_result notlinked
  cd -

fi

if [ $stage -le 28 ]; then
  # Evaluate the linked performance
  init_intra_thresh=-8.5
  init_show_thresh=38
  linked_start=$adapt_cluster_start
  scoring=plda_adapt

  cd $base
  rm -rf $adapt_result
  base-diar/run/step-cpdaic-kmedoids $task plda_nnet_adapt 20 $adapt_nnetdir/xvectors_$task $aic_path $adapt_purify_cluster
  base-diar/run/step-eval-kmedoids $task 10 $adapt_purify_cluster $adapt_result notlinked
  cd -

  cd $base
  base-diar/run/step-ahc-mat -THRESH $init_intra_thresh $task $scoring $adapt_nnetdir/xvectors_$task $linked_start $adapt_cluster_result/final
  base-diar/run/step-eval-cluster $task $adapt_cluster_result/final $adapt_result/final notlinked
  cd -

  mkdir -p $adapt_nnetdir/xvectors_${task}/adapt/final
  sid/ivector_nnet/create_semisupervised_data.sh $task $adapt_cluster_result/final $data/${task} $data_adapt/${task}_final
  utils/apply_map.pl -f 1 $data_adapt/${task}_final/utt_map < $adapt_nnetdir/xvectors_${task}/xvector.scp > $adapt_nnetdir/xvectors_${task}/adapt/final/xvector.scp

  # Clustering within shows using the episode clustering results.
  for show in `cut -d '-' -f 1 ${base}/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
    rm -rf $adapt_nnetdir/xvectors_${task}/adapt/final/$show $data_adapt/${task}_final/$show
    mkdir -p $data_adapt/${task}_final/$show $adapt_nnetdir/xvectors_${task}/adapt/final/$show
    grep `echo $show` $data_adapt/${task}_final/spk2utt > $data_adapt/${task}_final/$show/spk2utt
    grep `echo $show` $adapt_nnetdir/xvectors_${task}/adapt/final/xvector.scp > $adapt_nnetdir/xvectors_${task}/adapt/final/$show/xvector.scp

    # adapt_lambda is only used when plda_adapt is applied as the scoring method.
    sid/ivector_nnet/speaker_pairwise_score.sh --cmd "$train_cmd" --adapt-lambda "$adapt_nnetdir/xvectors_dev15" \
      $scoring $adapt_nnetdir/xvectors_voxceleb_train $data_adapt/${task}_final/$show $adapt_nnetdir/xvectors_${task}/adapt/final/$show &
  done
  wait

  for show in `cut -d '-' -f 1 ${base}/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
    $train_cmd $adapt_nnetdir/xvectors_${task}/adapt/final/$show/log/spk_clustering.log \
      python base-diar/python/spk_ahc_cluster_mat.py -p $show -t $init_show_thresh $adapt_nnetdir/xvectors_${task}/adapt/final/$show/score_${scoring}.txt $adapt_nnetdir/xvectors_${task}/adapt/final/$show/spk_ahc_${scoring} &
  done
  wait

  rm -f $adapt_nnetdir/xvectors_${task}/adapt/final/show_spk_ahc_${scoring}
  for show in `cut -d '-' -f 1 ${base}/lib/flists.test/${task}.lst | cut -d '_' -f 2 | sort -u`; do
    cat $adapt_nnetdir/xvectors_${task}/adapt/final/$show/spk_ahc_${scoring}
  done >> $adapt_nnetdir/xvectors_${task}/adapt/final/show_spk_ahc_${scoring}

  sid/ivector_nnet/speaker_cluster_data.sh $adapt_nnetdir/xvectors_${task}/adapt/final/show_spk_ahc_${scoring} $data_adapt/${task}_final $data_adapt/${task}_final_show_ahc

  cd $base
  base-diar/run/step-eval-spk-cluster $task $data_adapt/${task}_final_show_ahc/utt2spk $adapt_result/final_show_ahc notlinked
  base-diar/run/step-eval-spk-cluster $task $data_adapt/${task}_final_show_ahc/utt2spk $adapt_result/final_show_ahc linked
  cd -
fi


if [ $stage -le 29 ]; then
  # Re-train PLDA without fine-tune the network
  train_set=train
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 100 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data_adapt/${train_set}_0_show_ahc $nnetdir/xvectors_${train_set}_0_show_ahc

  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnetdir/xvectors_${train_set}_0_show_ahc/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_${train_set}_0_show_ahc/xvector.scp \
    $nnetdir/xvectors_${train_set}_0_show_ahc/mean.vec || exit 1;

  $train_cmd $nnetdir/xvectors_${train_set}_0_show_ahc/log/plda.log \
    ivector-compute-plda ark:$data_adapt/${train_set}_0_show_ahc/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_${train_set}_0_show_ahc/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnetdir/xvectors_${train_set}_0_show_ahc/plda || exit 1;

  for show in `cat $base/lib/flists.test/$task.lst`; do
    if [ -s $nnetdir/xvectors_$task/$show/trials ]; then
      $train_cmd $nnetdir/xvectors_$task/$show/log/plda_scoring_clust_adapt.log \
        ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:$nnetdir/xvectors_$task/$show/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_${train_set}_0_show_ahc/plda - |" \
        "ark:ivector-subtract-global-mean $nnetdir/xvectors_${train_set}_0_show_ahc/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $nnetdir/xvectors_${train_set}_0_show_ahc/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $nnetdir/xvectors_$task/$show/trials $nnetdir/xvectors_$task/$show/score_plda_clust_adapt.txt &
    fi
  done
  wait

  cd $base
  base-diar/run/step-score-eer $task plda_clust_adapt $nnetdir/xvectors_$task $aic_path
  base-diar/run/step-score-cluster-eer $task plda_clust_adapt normal normal $nnetdir/xvectors_$task $aic_path
  base-diar/run/step-score-cluster-eer $task plda_clust_adapt center center $nnetdir/xvectors_$task $aic_path
  cd -

  # Use different scoring to refine the AIC result.
  cd $base
  rm -rf $adapt_result
  base-diar/run/step-cpdaic-kmedoids $task plda_adapt 20 $nnetdir/xvectors_$task $aic_path $adapt_purify_cluster
  base-diar/run/step-eval-kmedoids $task 10 $adapt_purify_cluster $adapt_result notlinked
  cd -

  cd $base
  rm -rf $adapt_result
  base-diar/run/step-cpdaic-kmedoids $task plda_clust_adapt 20 $nnetdir/xvectors_$task $aic_path $adapt_purify_cluster
  base-diar/run/step-eval-kmedoids $task 10 $adapt_purify_cluster $adapt_result notlinked
  cd -

  # Use different scoring to do the clustering.
  cd $base
  base-diar/run/ahc-full-thresh-mat -4.0 0.5 -2.0 $task plda $nnetdir/xvectors_$task $adapt_cluster_start $adapt_cluster_result $adapt_result notlinked
  cd -

  cd $base
  base-diar/run/ahc-full-thresh-mat -2.0 0.5 2.0 $task plda_clust_adapt $nnetdir/xvectors_$task $adapt_cluster_start $adapt_cluster_result $adapt_result notlinked
  cd -

  cd $base
  base-diar/run/ahc-full-thresh-mat -8.0 0.5 -6.0 $task plda_adapt $nnetdir/xvectors_$task $adapt_cluster_start $adapt_cluster_result $adapt_result notlinked
  cd -
  exit 1
fi
