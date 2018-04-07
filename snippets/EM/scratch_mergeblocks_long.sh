#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=30000
#SBATCH --time=03:10:00
#SBATCH --job-name=EM_mergeblocks$ipf.$ids
export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2:$PATH
source activate root
export PYTHONPATH=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/pyDM3reader


ipf=_probs_eed
ids=sum16_eed
opf=${ipf}_sum16
ods=$ipf

for x in `seq 1000 $xs $(( xmax-1 ))`; do
X=$( get_coords_upper $x $xm $xs $xmax)
x=$( get_coords_lower $x $xm )
xrange=`printf %05d $x`-`printf %05d $X`
python /home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/wmem/mergeblocks.py $datadir/blocks_0500/${dataset}_${xrange}_00000-00520_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_00480-01020_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_00980-01520_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_01480-02020_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_01980-02520_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_02480-03020_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_02980-03520_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_03480-04020_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_03980-04520_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_04480-05020_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_04980-05520_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_05480-06020_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_05980-06520_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_06480-07020_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_06980-07520_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_07480-08020_00000-00184$ipf.h5/$ids $datadir/blocks_0500/${dataset}_${xrange}_07980-08316_00000-00184$ipf.h5/$ids $datadir/${dataset}$opf.h5/$ods -b 0 0 0 -p 184 500 500 -q 0 20 20 -s 184 8316 8423 -d float16
done
