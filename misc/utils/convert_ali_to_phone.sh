#!/bin/bash

cmd=run.pl
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

dir=$1
lang=$2

nj=`cat $dir/num_jobs`

# Generate monophones.txt and phones_filtered.txt manually

for id in $(seq $nj); do gunzip -c $dir/ali.$id.gz; done | \
  ali-to-phones --per-frame $dir/final.mdl ark:- ark,scp:$dir/phone.ark,$dir/phone.scp

python $TF_KALDI_ROOT/misc/utils/phones-to-monophones.py $lang/phones_filtered.txt $dir/phone.scp "ark:| copy-int-vector ark:- ark,scp:$dir/monophone.ark,$dir/monophone.scp"

