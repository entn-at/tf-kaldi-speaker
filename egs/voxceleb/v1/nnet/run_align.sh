#!/bin/bash
# Copyright 2012  Brno University of Technology (Author: Karel Vesely)
#           2013  Johns Hopkins University (Author: Daniel Povey)
#           2015  Vijayaditya Peddinti
#           2016  Vimal Manohar
# Apache 2.0

# Computes training alignments using nnet3 DNN

# Begin configuration section.
nj=32
use_gpu=false
cmd="run.pl"
checkpoint="last"
env=tf_cpu
per_frame_loglikes=
log_posteriors=

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
transform_dir=
chunk_size=10000
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: $0 [--transform-dir <transform-dir>] <transition-dir> <lang> <nnetdir> <data> <output-dir>"
   echo "e.g.: $0 exp/trans data/lang exp/nnet data exp/nnet/data"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

transdir=$1
lang=$2
nnetdir=$3
data=$4
dir=$5

rm -f $dir/ali.*.gz $dir/graph.*.fst

for f in $transdir/tree $nnetdir/nnet/checkpoint $data/feats.scp $lang/L.fst; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log
echo $nj > $dir/num_jobs
utils/split_data.sh $data $nj
sdata=$data/split$nj/JOB

oov=`cat $lang/oov.int` || exit 1;
tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/text|";
cp $transdir/tree $dir || exit 1;
cp $lang/phones.txt $dir || exit 1;

feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- |"
ali_wspecifier="ark:|gzip -c >$dir/ali.JOB.gz"

write_loglikes=
if [ ! -z $per_frame_loglikes ]; then
  loglikes_dir=`dirname $per_frame_loglikes`
  mkdir -p $loglikes_dir
  write_loglikes="--write-per-frame-acoustic-loglikes='ark,scp:${loglikes_dir}/per_frame_loglikes.JOB.ark,${loglikes_dir}/per_frame_loglikes.JOB.scp'"
fi

echo "$0: aligning data in $data using model from $nnetdir, putting alignments in $dir"

# Generate the alignment FST
$cmd JOB=1:$nj $dir/log/compile_graph.JOB.log \
  compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $transdir/tree $transdir/final.trans_mdl $lang/L.fst "$tra" ark:$dir/graph.JOB.fst

if $use_gpu; then
    echo "Using CPU to do inference is a better choice."
    exit 1
else
    source $TF_ENV/$env/bin/activate
    export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH
    python nnet/lib/make_checkpoint.py --checkpoint $checkpoint "$nnetdir"
    export PYTHONPATH=`pwd`/../../:$PYTHONPATH

    if [ ! -z $log_posteriors ]; then
        logpost_dir=`dirname $log_posteriors`
        mkdir -p $logpost_dir
        # It is pretty cool if we can feed the log-likelihood directly into the lattice decoder.
        $cmd JOB=1:$nj ${dir}/log/align.JOB.log \
        python nnet/lib/compute_loglike.py --gpu -1 --chunk-size $chunk_size \
          --write-per-frame-log-posteriors "ark:| copy-matrix ark:- ark,scp:${logpost_dir}/per_frame_logpost.JOB.ark,${logpost_dir}/per_frame_logpost.JOB.scp" \
          $transdir/prior.vec \
          "$nnetdir" \
          "$feat" \
          "ark:| align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam $write_loglikes \
                   $transdir/final.trans_mdl ark:$dir/graph.JOB.fst ark:- \"$ali_wspecifier\""
    else
        $cmd JOB=1:$nj ${dir}/log/align.JOB.log \
        python nnet/lib/compute_loglike.py --gpu -1 --chunk-size $chunk_size \
          $transdir/prior.vec \
          "$nnetdir" \
          "$feat" \
          "ark:| align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam $write_loglikes \
                   $transdir/final.trans_mdl ark:$dir/graph.JOB.fst ark:- \"$ali_wspecifier\""
    fi

    deactivate
fi

if [ ! -z $per_frame_loglikes ]; then
  for n in $(seq $nj); do
    cat ${loglikes_dir}/per_frame_loglikes.$n.scp || exit 1;
  done > $per_frame_loglikes
fi

if [ ! -z $log_posteriors ]; then
  for n in $(seq $nj); do
    cat ${logpost_dir}/per_frame_logpost.$n.scp || exit 1;
  done > $log_posteriors
fi

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir
echo "$0: done aligning data."
