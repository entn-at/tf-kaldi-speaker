#!/bin/bash

# Begin configuration section.
nj=4
cmd=run.pl
cmn_window=300
length_tolerance=2
compress=true
norm_vars=false

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

phone_class=$1
data=$2
bnf_dir=$3
post_dir=$4
data_out=$5
dir=$6

name=`basename $data`
mkdir -p $dir/log
mkdir -p $data_out

cp $data/utt2spk $data_out/utt2spk
cp $data/spk2utt $data_out/spk2utt
cp $data/wav.scp $data_out/wav.scp

featdir=$(utils/make_absolute.sh $dir)
write_num_frames_opt="--write-num-frames=ark,t:$featdir/log/utt2num_frames.JOB"

utils/split_data.sh $data $nj || exit 1;
sdata_in=$data/split$nj;

posts="scp:utils/filter_scp.pl ${sdata_in}/JOB/feats.scp ${post_dir}/post.scp |"
$cmd JOB=1:$nj $dir/log/convert_post.JOB.log \
  python misc/utils/convert_post.py $phone_class "$posts" "ark:| copy-matrix ark:- ark,scp:$dir/post.JOB.ark,$dir/post.JOB.scp"

acoustic_feats="ark:apply-cmvn-sliding --norm-vars=$norm_vars --center=true --cmn-window=$cmn_window scp:${sdata_in}/JOB/feats.scp ark:- |"
bn_feats="ark,s,cs:utils/filter_scp.pl ${sdata_in}/JOB/feats.scp ${bnf_dir}/feats.scp | apply-cmvn-sliding --norm-vars=$norm_vars --center=true --cmn-window=$cmn_window scp:- ark:- |"
posts_new="scp,s,cs:$dir/post.JOB.scp"

$cmd JOB=1:$nj $dir/log/create_xvector_feats_${name}.JOB.log \
  paste-feats --length-tolerance=$length_tolerance \
    "$acoustic_feats" "$bn_feats" "$posts_new" ark:- \| \
    select-voiced-frames ark:- scp,s,cs:${sdata_in}/JOB/vad.scp ark:- \| \
    copy-feats --compress=$compress $write_num_frames_opt ark:- \
    ark,scp:$featdir/xvector_feats_${name}.JOB.ark,$featdir/xvector_feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/xvector_feats_${name}.$n.scp || exit 1;
done > ${data_out}/feats.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/log/utt2num_frames.$n || exit 1;
done > $data_out/utt2num_frames || exit 1
rm $featdir/log/utt2num_frames.*

utils/fix_data_dir.sh $data_out

echo "Succeeded extracting features for $name into $data_out"

exit 0
