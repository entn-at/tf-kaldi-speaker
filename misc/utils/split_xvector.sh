#!/bin/bash

cmd=run.pl
nj=40

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

start_dim=$1
end_dim=$2
xvector_in=$3
xvector_out=$4

mkdir -p $xvector_out
mkdir -p $xvector_in/split$nj

vectors=$(for n in `seq $nj`; do echo $xvector_in/split${nj}/$n/xvector.scp; done)
directories=$(for n in `seq $nj`; do echo $xvector_in/split${nj}/$n; done)
mkdir -p $directories

utils/split_scp.pl $xvector_in/xvector.scp $vectors

$cmd JOB=1:$nj $xvector_out/log/create_subvector.JOB.log \
  copy-vector-dim-range --start-dim=$start_dim --end-dim=$end_dim \
    scp:$xvector_in/split$nj/JOB/xvector.scp ark,scp:$xvector_out/xvector.JOB.ark,$xvector_out/xvector.JOB.scp

for n in $(seq $nj); do
  cat $xvector_out/xvector.$n.scp || exit 1;
done > $xvector_out/xvector.scp || exit 1;
