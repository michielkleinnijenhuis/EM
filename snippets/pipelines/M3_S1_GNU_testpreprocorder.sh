local_datadir=/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU
scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU
mkdir -p $datadir && cd $datadir
module load python/2.7
z=0
Z=460

### register, then stitch ###

# register slices of the montages
for montage in m001 m002 m003; do
# register one 
sed "s?SOURCE_DIR?$datadir/tifs_$montage?;\
    s?TARGET_DIR?$datadir/reg_$montage?;\
    s?REFNAME?0000_$montage.tif?g" \
    $scriptdir/EM_register.py \
    > $datadir/EM_register_$montage.py
done

qsubfile=$datadir/EM_ppmon.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=10:00:00" >> $qsubfile
echo "#PBS -N em_ppmon" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "mon=m\`printf %03d \$PBS_ARRAYID\`" >> $qsubfile
echo "mkdir $datadir/tifs_\$mon" >> $qsubfile
echo "mkdir -p $datadir/reg_\$mon/trans" >> $qsubfile
echo "cp $datadir/tifs/????_\$mon.tif $datadir/tifs_\$mon/" >> $qsubfile
echo "/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64 --headless $datadir/EM_register_\$mon.py" >> $qsubfile
echo "python $scriptdir/EM_series2stack.py \
$datadir/reg_\$mon $datadir/reg_\$mon.h5 \
-f 'stack' -o -e 0.0073 0.0073 0.05" >> $qsubfile
echo "python $scriptdir/EM_stack2stack.py \
$datadir/reg_\$mon.h5 $datadir/training_\$mon.h5 \
-x 2000 -X 2500 -y 1000 -Y 1500 -z 250 -Z 350 -m .nii" >> $qsubfile
echo "mkdir -p $datadir/reg_\${mon}_ds" >> $qsubfile
echo "python $scriptdir/EM_downsample.py -i $datadir/reg_\$mon -o $datadir/reg_\${mon}_ds -d 10 -z $z -Z $Z -p _\$mon" >> $qsubfile
echo "python $scriptdir/EM_series2stack.py \
$datadir/reg_\${mon}_ds $datadir/reg_\${mon}_ds.h5 \
-f 'stack' -o -e 0.073 0.073 0.05" >> $qsubfile
qsub -t 0-3 $qsubfile


#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/training_m00* $local_datadir
#rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:$datadir/reg_m00?_ds.h5 $local_datadir
# inspect the data
# FIXME: 2D downsampling function
# try to do stitching

for montage in 000 001 002 003; do
sed "s?INPUTDIR?$datadir/reg_m$montage/?;\
    s?OUTPUTFILE?$datadir/reg_m$montage.ome.tif?g" \
    $datadir/EM_h2t.py > \
    $datadir/EM_h2t_m$montage.py
done

qsubfile=$datadir/EM_h2t.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_h2t" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "mon=m\`printf %03d \$PBS_ARRAYID\`" >> $qsubfile
echo "/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64 --headless $datadir/EM_h2t_\$mon.py" >> $qsubfile
qsub -q develq -t 0-3 $qsubfile


sed "s?INPUTDIR?$datadir/?;\
    s?OUTPUTDIR?$datadir/?g" \
    $datadir/EM_r2s.py > \
    $datadir/EM_r2s_s.py

qsubfile=$datadir/EM_r2s_s.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=10:00:00" >> $qsubfile
echo "#PBS -N em_r2s_s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64 --headless $datadir/EM_r2s_s.py" >> $qsubfile
qsub $qsubfile


#!/usr/bin/env python

"""
Stitch tiles in a montage for a range of slices.
"""
import sys
from os import path
from ij import IJ
from loci.plugins import BF
def main(argv):
    inputdir = 'INPUTDIR'
    outputdir = 'OUTPUTDIR'
    IJ.run("Grid/Collection stitching", 
           "type=[Grid: row-by-row] order=[Right & Down                ] \
           grid_size_x=2 \
           grid_size_y=2 \
           tile_overlap=10 \
           first_file_index_i=0 \
           directory=" + inputdir + " \
           file_names=reg_m{iii}.ome.tif \
           output_textfile_name=TileConfiguration.txt \
           fusion_method=[Linear Blending] \
           regression_threshold=0.30 \
           max/avg_displacement_threshold=2.50 \
           absolute_displacement_threshold=3.50 \
           compute_overlap \
           subpixel_accuracy \
           image_output=[Fuse and display]")
    IJ.saveAs("Tiff", path.join(outputdir, "reg_stitched.tif"))
if __name__ == "__main__":
    main(sys.argv[1:])



# continue with stitched stack



