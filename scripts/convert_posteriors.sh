#!/bin/bash

cmd="run.pl"
dataset=

echo "$0 $@"
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 [options] <phone-set> <gmm-dir> <phones.txt> <post-input-dir> <post-output-dir>"
  echo "Options:"
  echo "  --cmd <run.pl>"
  exit 100
fi

phone_set=$1
gmmdir=$2
phones=$3
post_in=$4
post_out=$5

suffix=
if [ ! -z $dataset ]; then
  suffix="_$dataset"
fi

mkdir -p ${post_out}/log
nj=`cat ${post_in}/num_jobs` || exit 1;

transition_id-to-pdf_id $gmmdir/trans/final.trans_mdl > $post_out/transition_id-to-pdf_id.txt

export OMP_NUM_THREADS=1

$cmd JOB=1:$nj ${post_out}/log/convert.JOB.log \
  python scripts/convert_posteriors.py $phone_set $post_out/transition_id-to-pdf_id.txt $phones \
    $post_in/post${suffix}.JOB.scp "ark:| copy-matrix ark:- ark,scp:$post_out/post${suffix}.JOB.ark,$post_out/post${suffix}.JOB.scp"

#python scripts/convert_posteriors.py $phone_set $post_out/transition_id-to-pdf_id.txt $phones \
#  $post_in/post${suffix}.1.scp "ark:| copy-matrix ark:- ark,scp:$post_out/post${suffix}.1.ark,$post_out/post${suffix}.1.scp"

for j in $(seq $nj); do cat $post_out/post${suffix}.$j.scp; done > $post_out/post.scp || exit 1;
