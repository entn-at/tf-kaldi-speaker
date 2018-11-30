#!/bin/bash

use_env=true
gpuid=-1
min_chunk_size=25
chunk_size=10000
normalize=false

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 [options] <conda-env> <nnet-dir> <data> <embeddings-dir>"
  echo "Options:"
  echo "  --gpuid <-1>"
  echo "  --min-chunk-size <25>"
  echo "  --chunk-size <10000>"
  echo "  --normalize <false>"
  echo ""
  exit 100
fi

env=$1
nnetdir=$2
feat=$3
dir=$4

if $use_env; then
  # If conda is used, set the environment and unset the predefined variables.
  source activate $env
  unset PYTHONPATH
  export LD_LIBRARY_PATH=/home/dawna/mgb3/transcription/exp-yl695/software/anaconda2/lib:$LD_LIBRARY_PATH
fi

if $normalize; then
  cmdopt_norm="--normalize"
fi

export PYTHONPATH=`pwd`/../../:$PYTHONPATH
python nnet/lib/extract.py --gpu $gpuid --min-chunk-size $min_chunk_size --chunk-size $chunk_size $cmdopt_norm \
         "$nnetdir" "$feat" "$dir"
