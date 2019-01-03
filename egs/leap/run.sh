#!/bin/bash

. ./cmd.sh
. ./path.sh

# Issues:
# 3. The format is different between MGB and LeaP? The position of the cluster indicator is different.
# 4. The number of clusters is really limited after AIC.

# The name of the task.
task=LPINT

# The kaldi egs directory is used to link some tools from kaldi.
# Voxceleb is used if the sample rate is 16K (SRE16 may be used for 8K).
kaldi_voxceleb=/home/dawna/mgb3/transcription/exp-yl695/software/kaldi_cpu/egs/voxceleb

# Kaldi directories for the experiment.
root_dir=/home/dawna/mgb3/transcription/exp-yl695/leap
mfccdir=$root_dir/mfcc
vaddir=$root_dir/mfcc
data=$root_dir/data
exp=$root_dir/exp

# The wave file directory
wavdir=/home/dawna/mjfg/LeaP/data/audio_16kHz

# The segmenter directory
segdir=/home/dawna/mgb3/transcription/exp-yl695/leap/decode-LPINT
convmap=/home/dawna/mjfg/LeaP/scoring/lib/confs/LPINT.convmap

# The sub-string of the aic directory. All shows share the same path (with different prefix).
# The final results is not under-clustered. The clustering must be from the scratch.
aic_path=decode/clust/cpdaic_2.5_3000/lib/flists
from_scratch=-ORG

# Similar to AIC result, the clustering result will be saved in scp file
cluster_result=decode/clust/cpdaic_2.5_3000/lib/flists.ahc

# The network used and the restored checkpoint.
# If checkpoint is -1, it means load the lastest model.
# In this case, the network is pre-trained with plda files.
nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2_m0.9
checkpoint=-1

# On MGB, if plda is used, the threshold is about -4.5 -- -2.5; if plda_adapt is used, it is about -8.5 -- -7.0
episode_thresh=-4.5

stage=8


if [ $stage -le -1 ]; then
    rm -fr utils steps sid conf local nnet base-diar

    # Link the directories from kaldi
    ln -s $kaldi_voxceleb/v2/utils ./
    ln -s $kaldi_voxceleb/v2/steps ./
    ln -s $kaldi_voxceleb/v2/sid ./
    ln -s $kaldi_voxceleb/v2/conf ./
    ln -s $kaldi_voxceleb/v2/local ./
    ln -s ../voxceleb/nnet ./

    # Also some tools in my directory.
    ln -s /home/dawna/mgb3/transcription/exp-yl695/base base-diar
    ln -s /home/dawna/mgb3/transcription/exp-yl695/base $segdir/base-diar
    exit 1
fi


if [ $stage -le 0 ]; then
  # Make data directory (in Kaldi format) from the CPD-AIC result.
  # Generate wav.scp, segments, utt2spk and spk2utt
  # To generate `wav.scp`, we need a coding file to map the wav file to the feature.
  mkdir -p $data/$task
  rm -f $data/$task/segments $data/$task/utt2spk

  # wav.scp: CLPINT-ABC1X-14032002-AB0000-en /home/dawna/mjfg/LeaP/data/audio_16kHz/ab_pol_eng_f_free_c1.wav
  awk '{print $2" '${wavdir}'/"$1".wav"}' $convmap > $data/$task/wav.scp

  # The awk may need to be slightly modified when applied to other datasets.
  for show in `cat $segdir/lib/flists.test/$task.lst`; do
    show_dir=$segdir/test/$task/${show}.1
    # spk2utt and utt2spk
    # Before:
    #   CLPINT-AWC1X-29072003-AW0000-en_C0001XX_0000078_0000702.plp=/home/dawna/alta/LeaP/data/plp/CLPINT-AWC1X-29072003-AW0000-COXXXXX.plp[78,702]
    # After:
    #   CLPINT-AWC1X-29072003-AW0000-en-0000078-0000702 CLPINT-AWC1X-29072003-AW0000-en-0000078-0000702
    cut -d '=' -f 1 $show_dir/$aic_path/${show}.scp | awk -F '[_.]' '{print $1"-"$(NF-2)"-"$(NF-1)}' | awk '{print $1" "$1}' >> $data/$task/utt2spk
    cp $data/$task/utt2spk $data/$task/spk2utt

    # segments: CLPINT-AWC1X-29072003-AW0000-en-0000078-0000702 CLPINT-AWC1X-29072003-AW0000-en 0.78 7.02
    cut -d ' ' -f 1 $data/$task/utt2spk | rev | cut -d '-' -f 3- | rev | paste -d ' ' - $data/$task/utt2spk |\
     awk -F '-' '{print $0" "$(NF-1)/100" "$(NF)/100}' | awk '{print $2" "$1" "$4" "$5}' >> $data/$task/segments
  done
  utils/fix_segments.sh $data/$task
  utils/fix_data_dir.sh $data/$task
fi


if [ $stage -le 1 ]; then
  # Extract features (for 16K wav)
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $data/$task $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data/$task
fi

if [ $stage -le 2 ]; then
  # Extract the embeddings (d-vector) without VAD (assume the segments are the output of segmenter).
  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj 40 --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false \
    $nnetdir $data/$task $nnetdir/xvectors_$task
fi

if [ $stage -le 3 ]; then
  # To use Kaldi script to do PLDA scoring, a trial list should be provide.
  # The pair-wise trial list is first generated for each episode.
  # The name of the episode does not consistent with the name of the feature, making it a little annoying to convert the name.
  for show in `cat $segdir/lib/flists.test/$task.lst`; do
    mkdir -p $nnetdir/xvectors_$task/$show
    grep `echo $show | cut -d '_' -f 2 | cut -d '-' -f 1,2` $nnetdir/xvectors_$task/xvector.scp > $nnetdir/xvectors_$task/$show/xvector.scp
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
  # PLDA scoring
  for show in `cat $segdir/lib/flists.test/$task.lst`; do
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

if [ $stage -le 5 ]; then
  # The unsupervised PLDA adaptation can be trianed on the test data
  $train_cmd $nnetdir/xvectors_$task/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_$task/xvector.scp \
      $nnetdir/xvectors_$task/mean.vec || exit 1;
  $train_cmd $nnetdir/xvectors_$task/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $nnetdir/xvectors_voxceleb_train/plda \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_$task/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_$task/plda_adapt || exit 1;
fi

if [ $stage -le 6 ]; then
  # PLDA scoring using the adapted PLDA
  for show in `cat $segdir/lib/flists.test/$task.lst`; do
    if [ -s $nnetdir/xvectors_$task/$show/trials ]; then
      $train_cmd $nnetdir/xvectors_$task/$show/log/plda_adapt_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:$nnetdir/xvectors_$task/$show/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_$task/plda_adapt - |" \
        "ark:ivector-subtract-global-mean $nnetdir/xvectors_$task/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $nnetdir/xvectors_$task/mean.vec scp:$nnetdir/xvectors_$task/$show/xvector.scp ark:- | transform-vec $nnetdir/xvectors_voxceleb_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $nnetdir/xvectors_$task/$show/trials $nnetdir/xvectors_$task/$show/score_plda_adapt.txt &
    fi
  done
  wait
fi

if [ $stage -le 7 ]; then
  # Clustering using the pairwise distance.
  # We may not want to use AIC result or do any AIC purification since the number of clusters after AIC is limited.
  # The clustering is from scratch, though the `-ORG` flag can be removed to use AIC result.
  # The scoring method is the suffix of the score files. For example, if we have score_plda.txt for each episode, the scoring method is `plda`.
  # The scoring method can also be `plda_adapt`.
  # The following scripts have to be run in the task directory.
  # The log files are in decode-LPINT/LOGs/LPINT/ahc_mat
  cd $segdir
#  # 1. Using a threshold to stop clustering.
#  base-diar/run/step-ahc-mat -leap -ORG -THRESH $episode_thresh $task plda $nnetdir/xvectors_$task $aic_path $cluster_result
  # 2. Keep clustering until 2 speakers left.
  base-diar/run/step-ahc-mat -leap -ORG -NUM 2 $task plda $nnetdir/xvectors_$task $aic_path $cluster_result
  cd -
fi

if [ $stage -le 8 ]; then
  mkdir -p $exp/score
  cat $segdir/test/LPINT/LPINT_CLPINT-*/decode/clust/cpdaic_2.5_3000/lib/flists.ahc/*.scp > $exp/score/result
  python eval_leap_stm.py $convmap $exp/score/result /home/dawna/mjfg/LeaP/scoring/lib/stms/LPINT.stm
  exit 1
fi


if [ $stage -le 9 ]; then
  # This step exhibits how to purify the AIC result (when it is under clustered).
  # The log files are in decode-LPINT/LOGs/LPINT/cpdaic_kmedoids/
  purify_cluster=decode/clust/cpdaic_2.5_3000/lib/flists.kmedoids
  iter=10
  cd $segdir
  base-diar/run/step-cpdaic-kmedoids -leap $task plda $iter $nnetdir/xvectors_$task $aic_path $purify_cluster
  base-diar/run/step-ahc-mat -NUM 2 $task plda $nnetdir/xvectors_$task $purify_cluster/$iter $cluster_result
  cd -
fi


