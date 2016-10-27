#!/bin/bash

# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#           2016       Feiteng
# Apache 2.0.

# This script does decoding with a nnet+ctc neural-net.  If the neural net was
# built on top of fMLLR transforms from a conventional system, you should
# provide the --transform-dir option.

# Begin configuration section.
stage=1
transform_dir=    # dir to find fMLLR transforms.
nj=3 # number of decoding jobs.  If --transform-dir set, must match that number!
acwt=1.0  # Just a default value, used for adaptation and beam-pruning..
blank_scale=1.0
shift=0 # frame shift.. for combination.
lattice_acoustic_scale=10.0  # This is kind of a hack; it's used for
                             # compatibility with existing scoring scripts,
                             # since we normally search the LM-scale in integer
                             # increments, and this doesn't work when the
                             # expected LM scale is around one.  We scale
                             # acoustic probs by 10 before writing them out; we
                             # expect that the tuned LM-scale after scoring will
                             # be close to 10.
blank_threshold=0.95 # bigger keep more frames for decoding

cmd=run.pl
extra_left_context=0

beam=15.0
lattice_beam=6.0 # Beam we use in lattice generation.
max_active=7000
min_active=200

ivector_scale=1.0
iter=final
scoring_opts="--min-lmwt 5 --max-lmwt 15"
skip_scoring=false
feat_type=
online_ivector_dir=
minimize=false
decode_opts=""

verbose=0
frame_shift=0

# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
  echo "e.g.:   steps/nnet3/ctc/decode.sh --nj 8 \\"
  echo "--online-ivector-dir exp/ctc/ivectors_test_eval92 \\"
  echo "    exp/ctc/nnet_tdnn_a/graph_tgpr data/test_eval92_hires exp/ctc/nnet_tdnn_a/decode_bg_eval92"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 15.0"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  echo "  --scoring-opts <string>                  # options to local/score.sh"
  exit 1;
fi

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
model="nnet-am-copy --remove-dropout=true $srcdir/$iter.mdl - |" 


[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $graphdir/CTC.fst $data/feats.scp $srcdir/$iter.mdl $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

frame_subsampling_factor=$(cat $srcdir/frame_subsampling_factor) || exit 1;

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
if [ -z "$feat_type" ]; then
  if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=raw; fi
  echo "$0: feature type is $feat_type"
fi

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)

  if [ $feat_type == "raw" ]; then trans=raw_trans;
  else trans=trans; fi
  if [ $feat_type == "lda" ] && \
    ! cmp $transform_dir/../final.mat $srcdir/final.mat && \
    ! cmp $transform_dir/final.mat $srcdir/final.mat; then
    echo "$0: LDA transforms differ between $srcdir and $transform_dir"
    exit 1;
  fi
  if [ ! -f $transform_dir/$trans.1 ]; then
    echo "$0: expected $transform_dir/$trans.1 to exist (--transform-dir option)"
    exit 1;
  fi
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    for n in $(seq $nj_orig); do cat $transform_dir/$trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/$trans.ark,$dir/$trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/$trans.scp ark:- ark:- |"
  else
    # number of jobs matches with alignment dir.
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/$trans.JOB ark:- ark:- |"
  fi
elif grep 'transform-feats --utt2spk' $srcdir/log/train.1.log >&/dev/null; then
  echo "$0: **WARNING**: you seem to be using a neural net system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi
##

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector_period=$ivector_period"
fi
if [ $shift != 0 ]; then
  feats="$feats shift-feats --shift=$shift ark:- ark:- |"
fi

if [ $stage -le 1 ]; then
  for n in $(seq $nj); do
    $cmd JOB=$n:$n $dir/log/decode.JOB.log \
      nnet2-ctc-latgen-faster --blank-threshold=$blank_threshold --verbose=$verbose $ivector_opts \
        --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam $decode_opts \
        --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
        --frame-subsampling-factor=$frame_subsampling_factor --frame-shift=$frame_shift \
        --word-symbol-table=$graphdir/words.txt "$model" \
        $graphdir/CTC.fst "$feats" \
        "ark:|lattice-scale --acoustic-scale=$lattice_acoustic_scale ark:- ark:- | gzip -c > $dir/lat.JOB.gz" || exit 1 &
    sleep 2
  done
  wait
fi

# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.

if [ $stage -le 2 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
  fi
fi
echo "Decoding done."
exit 0;
