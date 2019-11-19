#!/bin/bash

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi

if [ $# != 3 ]; then
  echo "$0 nnet-dir1 nnet-dir2 nnet-comb"
  exit 2
fi

dir1=$1
dir2=$2
dir=$3


mkdir -p $dir

python $TF_KALDI_ROOT/misc/utils/paste-vector.py $dir1/xvector.scp $dir2/xvector.scp "ark:| copy-vector ark:- ark,scp:$dir/xvector.ark,$dir/xvector.scp"
