#!/bin/bash

cmd="run.pl"
phone_set=

echo "$0 $@"
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 [options] <lang-dir> <gmm-dir> <ali-dir> <post-dir> <stat-dir>"
  echo "Options:"
  echo "  --cmd <run.pl>"
  echo "  --phone-set <data/lang/monophone.txt>"
  exit 100
fi

lang=$1
gmmdir=$2
alidir=$3
postdir=$4
dir=$5

mkdir -p ${dir}/log

nj=`cat $alidir/num_jobs` || exit 1;
nj_post=`cat $postdir/num_jobs` || exit 1;
[ $nj != $nj_post ] && echo "num_jobs of the ali and post directories should be the same" && exit 1

transition_id-to-pdf_id $gmmdir/trans/final.trans_mdl > $dir/transition_id-to-pdf_id.txt

cmdopt=
if [ ! -z $phone_set ]; then
  cmdopt="--phone-set $phone_set"
fi

# Note: the splits between the alignments and the posteriors are different.
# The alignments are split per-utt, while the posteriors are split per-speaker
# To make the splits the same, re-split the posteriors.
$cmd JOB=1:$nj ${dir}/log/extract_pdf.JOB.log \
  ali-to-pdf $gmmdir/trans/final.trans_mdl ark:"gunzip -c $alidir/ali.JOB.gz |" ark,scp:$alidir/ali_pdf.JOB.ark,$alidir/ali_pdf.JOB.scp

for n in `seq $nj`; do
  cat $alidir/ali_pdf.$n.scp
done > $alidir/ali_pdf.scp

export OMP_NUM_THREADS=1

$cmd JOB=1:$nj ${dir}/log/compute_stat.JOB.log \
  python scripts/diagnostic/compute_mean_posteriors.py $cmdopt $lang/phones.txt $dir/transition_id-to-pdf_id.txt \
    ark:$alidir/ali_pdf.JOB.ark \
    "scp:utils/filter_scp.pl $alidir/ali_pdf.JOB.scp $postdir/post.scp |" $dir/stat_posteriors.JOB.txt

stat_posts=$(for i in `seq $nj`; do echo $dir/stat_posteriors.$i.txt; done)
python scripts/diagnostic/combine_mean_posteriors.py $dir/stat_posteriors.txt $stat_posts
