#!/bin/bash

cmd="run.pl"
env=tf_gpu
num_gpus=1

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <valid-dir> <valid-spklist> <nnet>"
  echo "Options:"
  echo "  --env <tf_gpu>"
  exit 100
fi

valid=$1
valid_spklist=$2
nnetdir=$3

# add the library to the python path.
export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH
source $TF_ENV/$env/bin/activate

if [[ $env == "tf_gpu" ]]; then
  # Get available GPUs before we can train the network.
  num_total_gpus=`nvidia-smi -L | wc -l`
  num_gpus_assigned=0
  for i in `seq 0 $[$num_total_gpus-1]`; do
    # going over all GPUs and check if it is idle, and add to the list if yes
    if nvidia-smi -i $i | grep "No running processes found" >/dev/null; then
      num_gpus_assigned=$[$num_gpus_assigned+1]
    fi
    # once we have enough GPUs, break out of the loop
    [ $num_gpus_assigned -eq $num_gpus ] && break
  done
  [ $num_gpus_assigned -ne $num_gpus ] && echo "Not enough GPUs" && exit 1
  $cmd $nnetdir/log/train_nnet.log utils/parallel/limit_num_gpus.sh --num-gpus $num_gpus \
    python nnet/lib/train_insight.py $valid $valid_spklist $nnetdir
else
  python nnet/lib/train_insight.py $valid $valid_spklist $nnetdir
fi

deactivate

exit 0