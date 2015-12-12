# local:

scriptdir=/Users/michielk/workspace/EM_seg/src
oddir=/Users/michielk/oxdata/originaldata/P01/EM/M3/20Mar15/montage/Montage_
datadir=/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU

for montage in 000 001 002 003; do

mdir=$datadir/tifs_m$montage && mkdir -p $mdir

sed "s?INPUTDIR?$oddir$montage?;\
    s?OUTPUTDIR?$mdir?g" \
    $scriptdir/EM_convert2tif.py \
    > $datadir/EM_convert2tif_$montage.py
ImageJ --headless $datadir/EM_convert2tif_$montage.py

mpiexec -n 3 python $scriptdir/EM_convert2stack_blocks.py \
-i $mdir \
-o $datadir/test_data_m$montage.h5 \
-f 'stack' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 60

rm -rf $mdir

done


scriptdir=/Users/michielk/workspace/EM_seg/src
oddir=/Users/michielk/oxdata/originaldata/P01/EM/M3/20Mar15/montage/Montage_
datadir=/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU



montage=003
mdir=$datadir/tifs && mkdir -p $mdir
ImageJ --headless $datadir/EM_convert2tif_$montage.py



ImageJ --headless /Users/michielk/workspace/EM_seg/src/EM_montage2stitched.py
ImageJ --headless /Users/michielk/workspace/EM_seg/src/EM_register.py

