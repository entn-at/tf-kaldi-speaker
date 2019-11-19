#!/bin/bash

n=10
# alpha=0: use the posteriors, alpha=1: use the alignments, posterior_mapping=true: use the mapped posteriors
alpha=0
posterior_mapping=false
phone_set=

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 [options] <gmm-dir> <phones.txt> <ali-dir> <post-dir> <post-stat> <embedding-dir>"
  echo "Options:"
  echo "  --n <10>"
  echo "  --alpha <0>"
  echo "  --posterior-mapping <false>"
  echo "  --phone-set "
  echo ""
  exit 100
fi

gmmdir=$1
phones=$2
alidir=$3
postdir=$4
post_stat=$5
dir=$6

mkdir -p $dir
transition_id-to-pdf_id $gmmdir/trans/final.trans_mdl > $dir/transition_id-to-pdf_id.txt

cmdopt=""
if $posterior_mapping; then
  cmdopt="--stat $post_stat"
fi
if [ ! -z $phone_set ]; then
  cmdopt="$cmdopt --phone-set $phone_set"
fi

set -x
python scripts/diagnostic/compute_embedding_mean_variance.py $cmdopt -n $n --alpha $alpha \
  $dir/transition_id-to-pdf_id.txt $phones $alidir/ali_pdf.scp $postdir/post.scp $dir/xvector.scp
