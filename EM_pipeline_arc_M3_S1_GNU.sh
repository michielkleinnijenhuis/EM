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

module load python/2.7 mpi4py/1.3.1 hdf5-parallel/1.8.14_mvapich2

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

echo '#!/bin/bash' > $datadir/EM_series2stack_submit.sh
echo "#PBS -l nodes=1:ppn=16" >> $datadir/EM_series2stack_submit.sh
echo "#PBS -l walltime=01:00:00" >> $datadir/EM_series2stack_submit.sh
echo "#PBS -N em_s2s_ds" >> $datadir/EM_series2stack_submit.sh
echo "#PBS -V" >> $datadir/EM_series2stack_submit.sh
echo "cd \$PBS_O_WORKDIR" >> $datadir/EM_series2stack_submit.sh
echo ". enable_arcus_mpi.sh" >> $datadir/EM_series2stack_submit.sh
echo "mpirun \$MPI_HOSTS python $scriptdir/EM_series2stack.py \
'$datadir/reg_ds' \
'$datadir/reg_ds.h5' \
-f 'reg_ds' \
-o \
-e 0.073 0.073 0.05" >> $datadir/EM_series2stack_submit.sh

qsub $datadir/EM_series2stack_submit.sh

#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/reg_ds.h5 $local_datadir
