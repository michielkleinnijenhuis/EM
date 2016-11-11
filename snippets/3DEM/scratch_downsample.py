###==========================================###
### downsample registered slices for viewing ###
###==========================================###
# NOTE: ONLY WORKING ON ARCUS
module load python/2.7
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2

scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU

mkdir -p $datadir/m000_reg_ds

qsubfile=$datadir/EM_downsample_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_downsample.py \
'$datadir/m000_reg' '$datadir/m000_reg_ds' -d 10 -m" >> $qsubfile
qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_reg_ds/* /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_reg_ds_arcus/

qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_series2stack.py \
'$datadir/m000_reg_ds' '$datadir/m000_reg_ds.h5' \
-f 'stack' -m -o -e 0.073 0.073 0.05" >> $qsubfile
qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_reg_ds.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_reg_ds_arcus.h5






scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'

mkdir -p $datadir/m000_reg_arcus_ds

mpiexec -n 4 python $scriptdir/convert/EM_downsample.py \
$datadir/m000_reg_arcus $datadir/m000_reg_arcus_ds -d 10 -m

mpiexec -n 4 python $scriptdir/convert/EM_series2stack.py \
$datadir/tifs_ds $datadir/tifs_ds.h5 \
-f 'stack' -m -o -e 0.073 0.073 0.05
