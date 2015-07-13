###==========================###
### copy the data and ssh in ###
###==========================###
rsync -avz /Users/michielk/oxdata/originaldata/P01/EM/M3/17Feb15 ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3
ssh -Y ndcn0180@arcus.oerc.ox.ac.uk

###=====================###
### prepare environment ###
###=====================###

scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EMseg
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

###=====================###
### convert DM3 to tifs ###
###=====================###

mkdir -p $datadir/tifs

for montage in 000 001 002 003; do
sed "s?INPUTDIR?$oddir$montage?;\
    s?OUTPUTDIR?$datadir/tifs?;\
    s?OUTPUTPOSTFIX?m_$montage?g" \
    $scriptdir/EM_tiles2tif.py \
    > $datadir/EM_tiles2tif_m$montage.py
done

sed "s?DATADIR?$datadir?g" \
    $scriptdir/EM_tiles2tif_submit.sh \
    > $datadir/EM_tiles2tif_submit.sh

qsub $datadir/EM_tiles2tif_submit.sh

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

sed "s?INPUTDIR?$datadir/stitched?;\
    s?OUTPUTDIR?$datadir/reg?;\
    s?REFNAME?$reference_name?g" \
    $scriptdir/EM_register.py \
    > $datadir/EM_register.py

sed "s?DATADIR?$datadir?g" \
    $scriptdir/EM_register_submit.sh \
    > $datadir/EM_register.sh

qsub $datadir/EM_regsiter_submit.sh

