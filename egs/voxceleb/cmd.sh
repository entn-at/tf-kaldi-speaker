# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
#export train_cmd="queue.pl -l h=!air081 -l qp=low"
#export decode_cmd="queue.pl -l h=!air081 -l qp=low"
#export mkgraph_cmd="queue.pl -l h=!air081 -l qp=low"
#export cuda_cmd="queue.pl -l qp=cuda-low"

#export cuda_cmd="..."


#b) BUT cluster options
export train_cmd="queue.pl -l qp=low -l osrel='*' -l osrel='*'"
export decode_cmd="queue.pl -l qp=low -l osrel='*' -l osrel='*'"
export mkgraph_cmd="queue.pl -l qp=low -l osrel='*'"
export cuda_cmd="queue.pl -l qp=cuda-low -l osrel='*' -l gpuclass='*'"
# somehow, nnet-forward cpu usage exceeds 100% per job on air124

#c) run it locally...
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl