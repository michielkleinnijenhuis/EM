###==========================###
### copy the data and ssh in ###
###==========================###
local_datadir=/Users/michielk/oxdata/P01/EM/M3/M3_S1_SPL
local_oddir=/Users/michielk/oxdata/originaldata/P01/EM/M3/17Feb15
rsync -avz $local_oddir ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3
ssh -Y ndcn0180@arcus.oerc.ox.ac.uk

###=====================###
### prepare environment ###
###=====================###

scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
oddir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/17Feb15/montage/Montage_
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_SPL
reference_name=0000.tif
x=0
X=4000
y=0
Y=4000
z=0
Z=500
mkdir -p $datadir && cd $datadir

module load python/2.7

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

sed "s?INPUTDIR?$datadir/tifs?;\
    s?OUTPUTDIR?$datadir/stitched?;\
    s?Z_START?$z?;\
    s?Z_END?$Z?g" \
    $scriptdir/EM_montage2stitched.py \
    > $datadir/EM_montage2stitched.py

sed "s?DATADIR?$datadir?g" \
    $scriptdir/EM_montage2stitched_submit.sh \
    > $datadir/EM_montage2stitched_submit.sh

qsub $datadir/EM_montage2stitched_submit.sh

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
    s?DS_FACTOR?10?; \
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

