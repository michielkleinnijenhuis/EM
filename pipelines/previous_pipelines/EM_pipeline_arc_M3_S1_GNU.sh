###==========================###
### copy the data and ssh in ###
###==========================###
local_datadir=/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU
local_oddir=/Users/michielk/oxdata/originaldata/P01/EM/M3/20Mar15
#rsync -avz $local_oddir ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3
#ssh -Y ndcn0180@arcus.oerc.ox.ac.uk

rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/stitch_example_midstack $local_oddir/stitch_example_midstack

###=====================###
### prepare environment ###
###=====================###

scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
oddir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/20Mar15/montage/Montage_
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU
x=0
X=4000
y=0
Y=4000
z=0
Z=460

mkdir -p $datadir && cd $datadir

module load python/2.7 mpi4py/1.3.1 
module load hdf5-parallel/1.8.14_mvapich2
#module load hdf5-parallel/1.8.14_openmpi

###=====================###
### convert DM3 to tifs ###
###=====================###

mkdir -p $datadir/tifs

for montage in 000 001 002 003; do
sed "s?INPUTDIR?$oddir$montage?;\
    s?OUTPUTDIR?$datadir/tifs?;\
    s?OUTPUT_POSTFIX?_m$montage?g" \
    $scriptdir/EM_tiles2tif.py \
    > $datadir/EM_tiles2tif_m$montage.py
done

sed "s?DATADIR?$datadir?g" \
    $scriptdir/EM_tiles2tif_submit.sh \
    > $datadir/EM_tiles2tif_submit.sh

qsub -t 0-3 $datadir/EM_tiles2tif_submit.sh


###===================================###
### stitch and register slice montage ###
###===================================###

qsubfile=$datadir/EM_reg_getpairs_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=4:ppn=16" >> $qsubfile
echo "#PBS -l walltime=20:00:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/EM_reg_getpairs.py \
'$datadir/tifs' -o '$datadir/reg' \
-t 4 -c 1 -d 1 -k 10000 -f 0.1 0.1 -m -n 100 -r 1" >> $qsubfile
qsub $qsubfile
# n=6431

qsubfile=$datadir/EM_reg_optimize_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=16:ppn=16" >> $qsubfile
echo "#PBS -l walltime=10:00:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/EM_reg_optimize.py \
'$datadir/tifs' -o '$datadir/reg_d4' \
-u '$datadir/reg_d4/unique_pairs_c1_d4.pickle' \
-t 4 -c 1 -d 4 -i 50 -m" >> $qsubfile
qsub $qsubfile
# 2 iterations in 10 min using 8*16 nodes
# 10 iterations in 50 min using 16*16 nodes

qsubfile=$datadir/EM_reg_blend_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=8" >> $qsubfile
echo "#PBS -l walltime=01:00:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/EM_reg_blend.py \
'$datadir/tifs' '$datadir/reg_d4/betas_o1_d4.npy' '$datadir/reg_d4' \
-d 4 -i 1 -m" >> $qsubfile
qsub $qsubfile


rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4/*.tif $local_datadir/reg_d4/00tifs/




###==========================================###
### downsample registered slices for viewing ###
###==========================================###

qsubfile=$datadir/EM_downsample_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/EM_downsample.py \
'$datadir/reg_d4' '$datadir/reg_d4_ds' -d 4 -m" >> $qsubfile
qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/reg_ds $local_datadir/reg_ds



###====================================###
### convert image series to hdf5 stack ###
###====================================###

qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
echo "#PBS -l walltime=01:00:00" >> $qsubfile
echo "#PBS -N em_s2s_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/EM_series2stack.py \
'$datadir/reg_ds' '$datadir/reg_ds.h5' \
-f 'reg_ds' -m -o -e 0.073 0.073 0.05" >> $qsubfile
qsub $qsubfile
#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/reg_ds.h5 $local_datadir

###====================================###
### convert image series to hdf5 stack ###
###====================================###

qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=01:00:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python $scriptdir/EM_series2stack.py \
'$datadir/reg' '$datadir/reg.h5' \
-f 'reg' -o -e 0.0073 0.0073 0.05" >> $qsubfile
qsub $qsubfile

#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/reg.h5 $local_datadir


###===============================###
### Ilastik segmentation training ###
###===============================###

### create training dataset
qsubfile=$datadir/EM_stack2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python $scriptdir/EM_stack2stack.py \
$datadir/reg.h5 \
$datadir/training.h5 \
-f 'reg' -g 'stack' \
-e 0.05 0.0073 0.0073 \
-x 7000 -X 7500 -y 1000 -Y 1500 -z 250 -Z 350 -n" >> $qsubfile
#-x 7000 -X 7500 -y 2000 -Y 2500 -z 250 -Z 350 -n" >> $qsubfile
qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/training* $local_datadir

### perform interactive prediction
#...
#rsync -avz $local_datadir/training.ilp ndcn0180@arcus.oerc.ox.ac.uk:$datadir


###===================================###
### apply Ilastik classifier to stack ###
###===================================###

stack=training4
qsubfile=$datadir/EM_ac2s2.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_lc" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate ilastik-devel" >> $qsubfile
echo "CONDA_ROOT=\`conda info --root\`" >> $qsubfile
echo "\${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/training.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/${stack}_probabilities.h5 \
--output_internal_path=/probs \
$datadir/$stack.h5/stack" >> $qsubfile
qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/training4* $local_datadir


###====================###
### compute SLICvoxels ###
###====================###

qsubfile=$datadir/EM_slic.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_slic" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python $scriptdir/EM_slicvoxels.py \
-i $datadir/$stack.h5 \
-o $datadir/${stack}_slic.h5 \
-f 'stack' -g 'stack' -s 500" >> $qsubfile
echo "python $scriptdir/EM_slicvoxels.py \
-i $datadir/${stack}_probabilities.h5 \
-o $datadir/${stack}_probabilities_slic.h5 \
-f 'probs' -g 'stack' -s 500" >> $qsubfile
qsub -q develq $qsubfile










###===================================###
### apply Ilastik classifier to stack ###
###===================================###

qsubfile=$datadir/EM_applyclassifier2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=01:00:00" >> $qsubfile
echo "#PBS -N em_ac2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python ilastik.py --headless \
--project=$datadir/training.ilp \
--output_internal_path=/probs \
$datadir/reg.h5/reg" >> $qsubfile
qsub $qsubfile




qsubfile=$datadir/EM_stack2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python $scriptdir/EM_stack2stack.py \
$datadir/training_slic.h5 \
$datadir/training_slic.nii.gz \
-e 0.05 0.0073 0.0073 \
-d int32 -f stack -g stack" >> $qsubfile
qsub -q develq $qsubfile

#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/training_slic.nii.gz $local_datadir
