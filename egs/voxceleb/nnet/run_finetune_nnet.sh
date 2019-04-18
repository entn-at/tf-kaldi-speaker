#!/bin/bash

cmd="run.pl"
continue_training=false
env=tf_gpu
num_gpus=1
checkpoint=-1

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
  echo "Usage: $0 [options] <config> <train-dir> <train-spklist> <valid-dir> <valid-spklist> <pretrained-nnet> <nnet>"
  echo "Options:"
  echo "  --continue-training <false>"
  echo "  --checkpoint <-1>"
  echo "  --env <tf_gpu>"
  echo "  --num-gpus <n_gpus>"
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
export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH

mkdir -p $nnetdir/log

# Get available GPUs before we can train the network.
num_total_gpus=`nvidia-smi -L | wc -l`
num_gpus_assigned=0
while [ $num_gpus_assigned -ne $num_gpus ]; do
  num_gpus_assigned=0
  for i in `seq 0 $[$num_total_gpus-1]`; do
    # going over all GPUs and check if it is idle, and add to the list if yes
    if nvidia-smi -i $i | grep "No running processes found" >/dev/null; then
      num_gpus_assigned=$[$num_gpus_assigned+1]
    fi
    # once we have enough GPUs, break out of the loop
    [ $num_gpus_assigned -eq $num_gpus ] && break
  done
  [ $num_gpus_assigned -eq $num_gpus ] && break
  sleep 300
done

source $TF_ENV/$env/bin/activate
if [ $continue_training == 'true' ]; then
  # When continue training, just call the train.py
  $cmd $nnetdir/log/train_nnet.log \
    python nnet/lib/train.py -c --config $config $train $train_spklist $valid $valid_spklist $nnetdir
else
  $cmd $nnetdir/log/train_nnet.log \
    python nnet/lib/finetune.py --checkpoint $checkpoint --config $config $train $train_spklist $valid $valid_spklist $pretrain_nnetdir $nnetdir
fi
deactivate


source $TF_ENV/$env/bin/activate
$cmd $nnetdir/log/train_nnet.log utils/parallel/limit_num_gpus.sh --num-gpus $num_gpus \
    python nnet/lib/train.py $cmdopts --config $config $train $train_spklist $valid $valid_spklist $nnetdir
