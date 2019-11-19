#!/bin/bash

# Copyright 2016 Pegah Ghahremani

# This script dumps bottleneck feature for model trained using nnet3.
# CAUTION!  This script isn't very suitable for dumping features from recurrent
# architectures such as LSTMs, because it doesn't support setting the chunk size
# and left and right context.  (Those would have to be passed into nnet3-compute).
# See also chain/get_phone_post.sh.

# Begin configuration section.
stage=0
nj=4
cmd=run.pl
use_gpu=false
compress=true

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

node_name=$1
nnetdir=$2
data=$3
bnf_dir=$4
post_dir=$5

name=`basename $data`
sdata=$data/split$nj

mkdir -p $bnf_dir/log
mkdir -p $post_dir/log

split_data.sh $data $nj || exit 1;

feats="ark,s,cs:apply-cmvn-sliding --center=true scp:$sdata/JOB/feats.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: Generating bottleneck (BNF) features using $nnetdir model as output of "
  echo "    component-node with name $node_name."
  echo "output-node name=output input=$node_name" > $bnf_dir/output.config
  modified_bnf_nnet="nnet3-copy --nnet-config=$bnf_dir/output.config $nnetdir/final.raw - |"
  $cmd JOB=1:$nj $bnf_dir/log/make_bnf.JOB.log \
    nnet3-compute --use-gpu=no "$modified_bnf_nnet" "$feats" ark:- \| \
    copy-feats --compress=$compress ark:- ark,scp:$bnf_dir/raw_bnfeat.JOB.ark,$bnf_dir/raw_bnfeat.JOB.scp || exit 1;

  N0=$(cat $data/feats.scp | wc -l)
  N1=$(cat $bnf_dir/raw_bnfeat.*.scp | wc -l)
  if [[ "$N0" != "$N1" ]]; then
    echo "$0: Error generating BNF features (original:$N0 utterances, BNF:$N1 utterances)"
    exit 1;
  fi

  # Concatenate feats.scp into bnf_data
  for n in $(seq $nj); do  cat $bnf_dir/raw_bnfeat.$n.scp; done > $bnf_dir/feats.scp

  for f in segments spk2utt text utt2spk wav.scp vad.scp char.stm glm kws reco2file_and_channel stm; do
    [ -e $data/$f ] && cp -r $data/$f $bnf_dir/$f
  done

  echo "$0: done making BNF features."
  exit 0;
fi

if [ $stage -le 1 ]; then
  echo "$0: Generating posteriors using $nnetdir model"
  $cmd JOB=1:$nj $post_dir/log/make_post.JOB.log \
    nnet3-compute --use-gpu=no --apply-exp=true \
      $nnetdir/final.raw "$feats" ark:- \| \
      copy-matrix ark:- ark,scp:$post_dir/post.JOB.ark,$post_dir/post.JOB.scp || exit 1;

  N0=$(cat $data/feats.scp | wc -l)
  N1=$(cat $post_dir/post.*.scp | wc -l)
  if [[ "$N0" != "$N1" ]]; then
    echo "$0: Error generating posteriors (original:$N0 utterances, posteriors:$N1 utterances)"
    exit 1;
  fi

  # Concatenate feats.scp into bnf_data
  for n in $(seq $nj); do  cat $post_dir/post.$n.scp; done > $post_dir/post.scp

  echo "$0: done making posteriors."
fi

exit 0;

