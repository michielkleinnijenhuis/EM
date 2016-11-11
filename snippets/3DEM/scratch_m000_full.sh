# export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
# xmax=5217
# ymax=4460
# xs=1000; ys=1000;
# z=30; Z=460;
# datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"

### merge blocks to full m000 dataset
# pf=; field='stack'
# pf=_probs; field='volume/predictions'
pf=_probs0_eed2; field='stack'
q="d"
qsubfile=$datadir/EM_mb${pf}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --mem=25000" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py -i \
`ls $datadir/restored/${dataset}_?????-?????_?????-?????_?????-?????${pf}.h5 | tr "\n" " "` \
-o $datadir/${dataset}${pf}.h5 \
-f $field -l 'zyx' -b 0 0 30" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
### merge blocks to full m000 labels
# export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
pf=_labelMAmanedit; field='stack'
q="d"
qsubfile=$datadir/EM_mb${pf}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --mem=25000" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py -i \
`ls $datadir/restored/${dataset}_?????-?????_?????-?????_?????-?????${pf}.h5 | tr "\n" " "` \
-o $datadir/${dataset}${pf}.h5 \
-f $field -l 'zyx' -b 0 0 30 -r" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

# ### merge blocks to full m000 dataset (mpi4py on arcus-b: fails)
# # ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
# module load mpi4py/1.3.1
# module load hdf5-parallel/1.8.14_mvapich2_gcc
# module load python/2.7__gcc-4.8
# scriptdir="${HOME}/workspace/EM"
# datadir="${DATA}/EM/M3/M3_S1_GNU" && cd $datadir
# dataset='m000'
# ### merge all blocks
# # pf=; field='stack'
# pf=_probs; field='volume/predictions'
# # pf=_probs0_eed2; field='stack'  # no chunks!
# q="d"
# qsubfile=$datadir/EM_mb${pf}.sh
# echo '#!/bin/bash' > $qsubfile
# echo "#SBATCH --nodes=1" >> $qsubfile
# echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
# echo "#SBATCH --time=00:10:00" >> $qsubfile
# # echo "#SBATCH --mem=25000" >> $qsubfile
# echo "#SBATCH --job-name=EM_mb" >> $qsubfile
# echo ". enable_arcus-b_mpi.sh" >> $qsubfile
# echo "mpirun \$MPI_HOSTS python $scriptdir/convert/mergeblocks.py \
# $datadir/${dataset}${pf}_test.h5 -i \
# `ls $datadir/restored/${dataset}_?????-?????_?????-?????_?????-?????${pf}.h5 | tr "\n" " "` \
# -f ${field} -b 0 0 30 -m" >> $qsubfile
# [ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
# # ssh -Y ndcn0180@arcus.arc.ox.ac.uk
# # not having anaconda added to PATH, do:
# module load python/2.7 mpi4py/1.3.1 hdf5-parallel/1.8.14_mvapich2
# scriptdir="${HOME}/workspace/EM"
# datadir="${DATA}/EM/M3/M3_S1_GNU" && cd $datadir
# dataset='m000'
# ### merge all blocks (explicit)
# # pf=; field='stack'
# # pf=_probs; field='volume/predictions'
# pf=_probs0_eed2; field='stack'  # no chunks! not handled right!
# q="d"
# qsubfile=$datadir/EM_mb${pf}.sh
# echo '#!/bin/bash' > $qsubfile
# echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
# echo "#PBS -l walltime=00:10:00" >> $qsubfile
# echo "#PBS -N em_mb" >> $qsubfile
# echo "#PBS -V" >> $qsubfile
# echo "cd \$PBS_O_WORKDIR" >> $qsubfile
# echo ". enable_arcus_mpi.sh" >> $qsubfile
# echo "mpirun \$MPI_HOSTS python $scriptdir/convert/mergeblocks.py \
# $datadir/${dataset}${pf}_test.h5 -i \
# `ls $datadir/restored/${dataset}_?????-?????_?????-?????_?????-?????${pf}.h5 | tr "\n" " "` \
# -f ${field} -b 0 0 30 -m" >> $qsubfile
# [ "$q" = "d" ] && qsub -q develq $qsubfile || qsub $qsubfile


# masks for full dataset
q=""
qsubfile=$datadir/EM_prob2mask_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --mem-per-cpu=125000" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=02:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_p2m" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
$datadir $dataset -p \"\" stack -l 0 -u 10000000 -o _maskDS" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
$datadir $dataset -p _probs0_eed2 stack -l 0.2 -o _maskMM" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
$datadir $dataset -p _probs0_eed2 stack -l 0.02 -o _maskMM-0.02" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

### connected components in maskMM-0.02
q=""
qsubfile=$datadir/EM_conncomp_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --mem-per-cpu=125000" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "python $scriptdir/supervoxels/conn_comp.py \
$datadir $dataset --maskMM _maskMM-0.02 stack" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile



q="d"
qsubfile=$datadir/EM_nifti.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --mem-per-cpu=50000" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/m000_labelMAmanedit.h5 \
$datadir/m000_labelMAmanedit.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
