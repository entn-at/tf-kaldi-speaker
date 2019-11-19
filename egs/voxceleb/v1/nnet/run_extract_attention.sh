#!/bin/bash

nj=32
use_gpu=false
cmd="run.pl"
min_chunk_size=25
chunk_size=10000
stage=0
checkpoint=-1
env=tf_cpu

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <nnet-dir> <data> <attention-dir>"
  echo ""
  exit 100
fi

nnetdir=$1
data=$2
dir=$3

for f in $nnetdir/nnet/checkpoint $data/feats.scp $data/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting embeddings for $data"
sdata=$data/split$nj/JOB

feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"

if [ $stage -le 0 ]; then
  # Set the checkpoint.
  source $TF_ENV/$env/bin/activate
  export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH
  python nnet/lib/make_checkpoint.py --checkpoint $checkpoint "$nnetdir"
  deactivate

  if $use_gpu; then
    echo "Using CPU to do inference is a better choice."
    exit 1
  else
    if [ ! -z $env ]; then
      source $TF_ENV/$env/bin/activate
    fi
    export PYTHONPATH=`pwd`/../../:$PYTHONPATH

#    feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:$data/split$nj/1/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:$data/split$nj/1/vad.scp ark:- |"
#    python nnet/lib/extract_attention.py --gpu -1 --min-chunk-size $min_chunk_size --chunk-size $chunk_size \
#         "$nnetdir" "$feat" "ark:| copy-matrix ark:- ark,scp:${dir}/attention_weights.1.ark,${dir}/attention_weights.1.scp"

    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
      python nnet/lib/extract_attention.py --gpu -1 --min-chunk-size $min_chunk_size --chunk-size $chunk_size \
         "$nnetdir" "$feat" "ark:| copy-matrix ark:- ark,scp:${dir}/attention_weights.JOB.ark,${dir}/attention_weights.JOB.scp"

    deactivate

#      nnet/wrap/extract_wrapper.sh --gpuid -1 --env $env --min-chunk-size $min_chunk_size --chunk-size $chunk_size \
#        --normalize $normalize --node $node \
#        "$nnetdir" "$feat" "ark:| copy-vector ark:- ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp"
  fi
fi

if [ $stage -le 1 ]; then
  for j in $(seq $nj); do cat $dir/attention_weights.$j.scp; done >$dir/attention_weights.scp || exit 1;
fi

exit 0