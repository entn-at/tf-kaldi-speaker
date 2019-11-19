#!/bin/bash

cmd=run.pl
nj=40
compress=false
cmn_window=300
length_tolerance=5

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

data_in=$1
bnf_dir=$2
post_dir=$3
data_out=$4
dir=$5

name=`basename $data_in`

for f in $data_in/feats.scp $data_in/vad.scp $bnf_dir/feats.scp $post_dir/post.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

mkdir -p $dir/log
mkdir -p $data_out

featdir=$(utils/make_absolute.sh $dir)

cp $data_in/utt2spk $data_out/utt2spk
cp $data_in/spk2utt $data_out/spk2utt
cp $data_in/wav.scp $data_out/wav.scp

write_num_frames_opt="--write-num-frames=ark,t:$featdir/log/utt2num_frames.JOB"

utils/split_data.sh $data_in $nj || exit 1;
sdata_in=$data_in/split$nj;

## For data, cvmn-sliding
#$cmd JOB=1:$nj $dir/log/create_acoustic_feats_${name}.JOB.log \
#  apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window \
#  scp:${sdata_in}/JOB/feats.scp ark:- \| \
#  copy-feats $write_num_frames_opt ark:- \
#  ark,scp:$featdir/xvector_acoustic_feats_${name}.JOB.ark,$featdir/xvector_acoustic_feats_${name}.JOB.scp || exit 1;
#
## For bottleneck features, cmvn-sliding -> remove silence
#$cmd JOB=1:$nj $dir/log/create_bn_feats_${name}.JOB.log \
#  utils/filter_scp.pl ${sdata_in}/JOB/feats.scp ${bnf_dir}/feats.scp \| \
#  apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window scp:- ark:- \| \
#  copy-feats ark:- ark,scp:$featdir/xvector_bn_feats_${name}.JOB.ark,$featdir/xvector_bn_feats_${name}.JOB.scp || exit 1;
#
## For posteriors, remove silence
#$cmd JOB=1:$nj $dir/log/create_posts_${name}.JOB.log \
#  utils/filter_scp.pl ${sdata_in}/JOB/feats.scp ${post_dir}/post.scp \| \
#  copy-matrix scp:- ark,scp:$featdir/xvector_posts_${name}.JOB.ark,$featdir/xvector_posts_${name}.JOB.scp || exit 1;

# Combined in one command
acoustic_feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window scp:${sdata_in}/JOB/feats.scp ark:- |"
bn_feats="ark,s,cs:utils/filter_scp.pl ${sdata_in}/JOB/feats.scp ${bnf_dir}/feats.scp | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window scp:- ark:- |"
posts="scp,s,cs:utils/filter_scp.pl ${sdata_in}/JOB/feats.scp ${post_dir}/post.scp |"

$cmd JOB=1:$nj $dir/log/create_xvector_feats_${name}.JOB.log \
  paste-feats --length-tolerance=$length_tolerance \
    "$acoustic_feats" "$bn_feats" "$posts" ark:- \| \
    copy-feats --compress=$compress $write_num_frames_opt ark:- \
    ark,scp:$featdir/xvector_feats_${name}.JOB.ark,$featdir/xvector_feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/xvector_feats_${name}.$n.scp || exit 1;
done > ${data_out}/feats.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/log/utt2num_frames.$n || exit 1;
done > $data_out/utt2num_frames || exit 1
rm $featdir/log/utt2num_frames.*

nf=`cat ${data_out}/feats.scp | wc -l`
nu=`cat ${data_in}/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data_out"
fi

echo "Succeeded extracting features for $name into $data_out"
