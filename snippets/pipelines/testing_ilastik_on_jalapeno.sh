### local2jalapeno ###
DATA="$HOME/oxdata"
scriptdir="$HOME/workspace/EM"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'
dataset='m000_cutout01'

scp -r ${datadir}/pipeline_test/${pixprob_trainingset}.ilp jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU/
scp -r ${datadir}/pipeline_test/${pixprob_trainingset}.h5 jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU/
scp -r ${datadir}/${dataset}.h5 jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU/

### jalapeno2local ###
scp -r jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU/${dataset}_probs.h5 ${datadir}/${dataset}_probs_jalapeno.h5
scp -r jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU/${dataset}_eed2.h5 ${datadir}/${dataset}_eed2_jalapeno.h5




### jalapeno ###
DATA="/vols/Data/km/michielk"
scriptdir="$HOME/workspace/EM"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'
dataset='m000_cutout01'

source activate ilastik-devel
CONDA_ROOT=`conda info --root`
LAZYFLOW_THREADS=10 LAZYFLOW_TOTAL_RAM_MB=32000 ${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project="$datadir/${pixprob_trainingset}_jalapeno.ilp" \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format="${datadir}/${dataset}_probs.h5" \
--output_internal_path=/volume/predictions \
"${datadir}/${dataset}.h5/stack"

export PATH=/vols/Data/km/michielk/workspace/miniconda/bin:$PATH
source activate neuroproof-devel
neuroproof_graph_learn -h > /vols/Data/km/michielk/workspace/NeuroProof/examples/training_sample2/helpmessage_$1.txt
source activate neuroproof-devel
neuroproof_graph_learn -h > /vols/Data/km/michielk/workspace/NeuroProof/examples/training_sample2/helpmessage_$1.txt






### arcus-b ###
# scp -r ${datadir}/${dataset}.h5 ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/
# ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
# sacct --format=JobID,Timelimit,elapsed,ReqMem,MaxRss,AllocCPUs,AveVMSize,MaxVMSize,CPUTime,ExitCode | less
module unload intel-compilers/2013
module load intel-compilers/2015
module load python/2.7  # incompatibility??
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2
#module load hdf5-parallel/1.8.14_openmpi


source ~/.bashrc
scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'
dataset='m000_cutout01'



qsubfile=$datadir/EM_ac2s2.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_lc" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate ilastik-devel" >> $qsubfile
echo "CONDA_ROOT=\`conda info --root\`" >> $qsubfile
echo "\${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/${pixprob_trainingset}_arcus.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/${dataset}_probs.h5 \
--output_internal_path=/volume/predictions \
$datadir/$dataset.h5/stack" >> $qsubfile
sbatch -p compute $qsubfile
# sbatch -p devel $qsubfile
# qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_cutout01_probs.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_cutout01_probs_arcus.h5
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_probs.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_probs_arcus.h5




mkdir -p $datadir/m000_reg/trans

sed "s?SOURCE_DIR?$datadir/m000?;\
    s?TARGET_DIR?$datadir/m000_reg?;\
    s?REFNAME?0250_m000.tif?g" \
    $scriptdir/reg/EM_register.py \
    > $datadir/EM_register.py

qsubfile=$datadir/EM.reg_m000.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo "/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64 --headless $datadir/EM_register.py" >> $qsubfile
sbatch -p compute $qsubfile



###==========================================###
### downsample registered slices for viewing ###
###==========================================###

# FIXME??? NOT WORKING ON ARCUS-B
# start clean! (some incompatibilities on skimage with previous load)
source ~/.bashrc
module load intel-compilers/2015

# module load python/2.7  # incompatibility??
# module load mpi4py/1.3.1
# module load hdf5-parallel/1.8.14_mvapich2

scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/M3/M3_S1_GNU"

cd $datadir
mkdir -p $datadir/m000_reg_ds4

qsubfile=$datadir/EM_downsample_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ds" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_downsample.py \
'$datadir/m000_reg' '$datadir/m000_reg_ds4' -d 4 -m" >> $qsubfile
sbatch -p devel $qsubfile
#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/reg_ds $local_datadir/reg_ds




# NOTE: WORKING ON ARCUS

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







qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
echo "#PBS -l walltime=01:00:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_series2stack.py \
'$datadir/m000_reg' '$datadir/m000_reg.h5' \
-f 'stack' -m -o -e 0.0073 0.0073 0.05" >> $qsubfile
qsub $qsubfile
#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/reg_ds.h5 $local_datadir



