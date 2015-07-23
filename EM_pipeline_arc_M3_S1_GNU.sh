###==========================###
### copy the data and ssh in ###
###==========================###
local_datadir=/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU
local_oddir=/Users/michielk/oxdata/originaldata/P01/EM/M3/20Mar15
rsync -avz $local_oddir ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3
ssh -Y ndcn0180@arcus.oerc.ox.ac.uk

###=====================###
### prepare environment ###
###=====================###

scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
oddir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/20Mar15/montage/Montage_
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU
reference_name=0000.tif
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

###======================###
### stitch slice montage ###
###======================###

mkdir -p $datadir/stitched

jobstring=''
for slc in `seq 0 20 $((Z-20))`; do
sed "s?INPUTDIR?$datadir/tifs?;\
    s?OUTPUTDIR?$datadir/stitched?;\
    s?Z_START?$slc?;\
    s?Z_END?$((slc+20))?g" \
    $scriptdir/EM_montage2stitched.py \
    > $datadir/EM_montage2stitched_`printf %03d $slc`.py
    jobstring="$jobstring$slc,"
done

sed "s?DATADIR?$datadir?g" \
    $scriptdir/EM_montage2stitched_submit.sh \
    > $datadir/EM_montage2stitched_submit.sh

qsub -t ${jobstring%,} $datadir/EM_montage2stitched_submit.sh

###=================###
### register slices ###
###=================###

mkdir -p $datadir/reg/trans

sed "s?SOURCE_DIR?$datadir/stitched?;\
    s?TARGET_DIR?$datadir/reg?;\
    s?REFNAME?$reference_name?g" \
    $scriptdir/EM_register.py \
    > $datadir/EM_register.py

sed "s?DATADIR?$datadir?g" \
    $scriptdir/EM_register_submit.sh \
    > $datadir/EM_register_submit.sh

qsub $datadir/EM_register_submit.sh

###==========================================###
### downsample registered slices for viewing ###
###==========================================###

mkdir -p $datadir/reg_ds

sed "s?SCRIPTDIR?$scriptdir?;\
    s?INPUTDIR?$datadir/reg?;\
    s?OUTPUTDIR?$datadir/reg_ds?;\
    s?DS_FACTOR?10?;\
    s?X_START?$x?;\
    s?X_END?$X?;\
    s?Y_START?$y?;\
    s?Y_END?$Y?;\
    s?Z_START?$z?;\
    s?Z_END?$Z?g" \
    $scriptdir/EM_downsample_submit.sh \
    > $datadir/EM_downsample_submit.sh

qsub $datadir/EM_downsample_submit.sh

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
'$datadir/reg_ds' \
'$datadir/reg_ds.h5' \
-f 'reg_ds' \
-m \
-o \
-e 0.073 0.073 0.05" >> $qsubfile

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
'$datadir/reg' \
'$datadir/reg.h5' \
-f 'reg' \
-o \
-e 0.0073 0.0073 0.05" >> $qsubfile

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
-x 7000 -X 7500 -y 2000 -Y 2500 -z 250 -Z 350 -n" >> $qsubfile
qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/training* $local_datadir

### perform interactive prediction
#...
rsync -avz $local_datadir/pixclass.ilp ndcn0180@arcus.oerc.ox.ac.uk:$datadir

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
--project=$datadir/pixclass.ilp \
--output_internal_path=/probs \
$datadir/reg.h5/reg" >> $qsubfile

qsub $qsubfile



qsubfile=$datadir/EM_slic_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_slic" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python $scriptdir/EM_slicvoxels.py \
-i $datadir/training.h5 \
-o $datadir/training_slic.h5 \
-f 'stack' -g 'stack' -s 500" >> $qsubfile
qsub -q develq $qsubfile

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
