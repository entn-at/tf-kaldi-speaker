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

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting embeddings for $data"
sdata=$data/split$nj/JOB

# The data has been silence removed and wcmvn applied
if [ $checkpoint == "0" ]; then
  cmdopt="--init"
else
  source $TF_ENV/$env/bin/activate
  export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH
  python nnet/lib/make_checkpoint.py --checkpoint $checkpoint "$nnetdir"
  deactivate
fi

if $use_gpu; then
  echo "Using CPU to do inference is a better choice."
  exit 1
else
  if [ ! -z $env ]; then
    source $TF_ENV/$env/bin/activate
  fi
  export PYTHONPATH=`pwd`/../../:$PYTHONPATH

#python nnet/lib/extract_angle.py --gpu -1 --min-chunk-size $min_chunk_size --chunk-size $chunk_size \
#  "$nnetdir" "$data/split$nj/1/feats.scp" $data/utt2spk $data/spklist ${dir}/angles.1.scp

  $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
    python nnet/lib/extract_angle.py $cmdopt --gpu -1 --min-chunk-size $min_chunk_size --chunk-size $chunk_size \
       "$nnetdir" "$data/split$nj/JOB/feats.scp" $data/utt2spk $data/spklist ${dir}/angles.JOB.scp

  deactivate
fi

for j in $(seq $nj); do cat $dir/angles.$j.scp; done > $dir/angles.scp || exit 1;

exit 0
