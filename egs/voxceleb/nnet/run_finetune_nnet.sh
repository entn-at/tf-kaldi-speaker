#!/bin/bash

cmd="run.pl"
continue_training=false
checkpoint=-1

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
  echo "Usage: $0 [options] <config> <train-dir> <train-spklist> <valid-dir> <valid-spklist> <pretrained-nnet> <nnet>"
  echo "Options:"
  echo "  --continue-training <false>"
  echo "  --checkpoint <-1>"
  exit 100
fi

config=$1
train=$2
train_spklist=$3
valid=$4
valid_spklist=$5
pretrain_nnetdir=$6
nnetdir=$7

# add the library to the python path.
export PYTHONPATH=`pwd`/../../:$PYTHONPATH

mkdir -p $nnetdir/log

if [ $continue_training == 'true' ]; then
  # When continue training, just call the train.py
  $cmd $nnetdir/log/train_nnet.log \
    python nnet/lib/train.py -c --config $config $train $train_spklist $valid $valid_spklist $nnetdir
else
  $cmd $nnetdir/log/train_nnet.log \
    python nnet/lib/finetune.py --checkpoint $checkpoint --config $config $train $train_spklist $valid $valid_spklist $pretrain_nnetdir $nnetdir
fi
