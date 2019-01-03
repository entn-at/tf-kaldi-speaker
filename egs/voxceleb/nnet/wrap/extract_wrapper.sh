#!/bin/bash

env=
gpuid=-1
min_chunk_size=25
chunk_size=10000
normalize=false

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <nnet-dir> <data> <embeddings-dir>"
  echo "Options:"
  echo "  --gpuid <-1>"
  echo "  --min-chunk-size <25>"
  echo "  --chunk-size <10000>"
  echo "  --normalize <false>"
  echo ""
  exit 100
fi

nnetdir=$1
feat=$2
dir=$3

if [ ! -z $env ]; then
  # If conda is used, set the environment and unset the predefined variables.
#  source activate $env
  source $HOME/$env/bin/activate
  unset PYTHONPATH
  export LD_LIBRARY_PATH=/home/dawna/mgb3/transcription/exp-yl695/software/anaconda2/lib:$LD_LIBRARY_PATH
fi

if $normalize; then
  cmdopt_norm="--normalize"
fi

# Hardly set the MKL-related variables to make the code run on one cpu.
#export MKL_NUM_THREADS=1
#export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=1"
#export OMP_NUM_THREADS=1
# export MKL_DYNAMIC="FALSE"
# export OMP_DYNAMIC="FALSE"

export PYTHONPATH=`pwd`/../../:$PYTHONPATH

python nnet/lib/extract.py --gpu $gpuid --min-chunk-size $min_chunk_size --chunk-size $chunk_size $cmdopt_norm \
         "$nnetdir" "$feat" "$dir"
