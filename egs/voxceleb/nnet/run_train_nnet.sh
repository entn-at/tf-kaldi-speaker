#!/bin/bash

cmd="run.pl"
continue_training=false
env=tf_gpu

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 [options] <config> <train-dir> <train-spklist> <valid-dir> <valid-spklist> <nnet>"
  echo "Options:"
  echo "  --continue-training <false>"
  exit 100
fi

config=$1
train=$2
train_spklist=$3
valid=$4
valid_spklist=$5
nnetdir=$6

# add the library to the python path.
export PYTHONPATH=`pwd`/../../:$PYTHONPATH

mkdir -p $nnetdir/log

if [ $continue_training == 'true' ]; then
  cmdopts="-c"
fi

# Activate the gpu virtualenv
# The tensorflow is installed using pip (virtualenv). Modify the code if you activate TF by other ways.
source $TF_ENV/$env/bin/activate
$cmd $nnetdir/log/train_nnet.log \
    python nnet/lib/train.py $cmdopts --config $config $train $train_spklist $valid $valid_spklist $nnetdir
deactivate
