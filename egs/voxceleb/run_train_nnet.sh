#!/bin/bash

cmd=run.pl
continue_training=false

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 [options] <config> <train_dir> <train_spklist> <valid_dir> <valid_spklist> <nnet>"
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

$cmd $nnetdir/log/train_nnet.log \
    python train.py $cmdopts --config $config $train $train_spklist $valid $valid_spklist $nnetdir

