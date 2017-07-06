#!/usr/bin/env python

"""Stitch tiles in a montage.

ImageJ headless doesn't take arguments easily...
Example usage on Oxford's ARC arcus cluster:

scriptdir="${HOME}/workspace/EM"
imagej=/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64
datadir="${DATA}/EM/M3/M3_S1_GNU"

mkdir -p $datadir/stitched

i=0
for z in `seq 0 46 414`; do
Z=$((z+46))
sed "s?INPUTDIR?$datadir/tifs?;\
    s?OUTPUTDIR?$datadir/stitched?;\
    s?Z_START?$z?;\
    s?Z_END?$Z?g" \
    $scriptdir/reg/fiji_stitch.py \
    > $datadir/fiji_stitch_`printf %03d $i`.py
i=$((i+1))
done

qsubfile=$datadir/fiji_stitch_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=01:00:00" >> $qsubfile
echo "#PBS -N em_m2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "$imagej --headless \\" >> $qsubfile
echo "$datadir/fiji_stitch_\`printf %03d \$PBS_ARRAYID\`.py" >> $qsubfile
qsub -t 0-9 $qsubfile
"""

import sys
import os

from ij import IJ
from loci.plugins import BF


def main(argv):
    """Stitch tiles in a montage."""

    inputdir = 'INPUTDIR'
    outputdir = 'OUTPUTDIR'
    z_start = Z_START
    z_end = Z_END

    for slc in range(z_start, z_end):
        IJ.run("Grid/Collection stitching",
               "type=[Grid: row-by-row] order=[Right & Down                ] \
               grid_size_x=2 \
               grid_size_y=2 \
               tile_overlap=10 \
               first_file_index_i=0 \
               directory=" + inputdir + " \
               file_names=" + str(slc).zfill(4) + "_m{iii}.tif \
               output_textfile_name=" + str(slc).zfill(4) + "TileConfiguration.txt \
               fusion_method=[Linear Blending] \
               regression_threshold=0.30 \
               max/avg_displacement_threshold=2.50 \
               absolute_displacement_threshold=3.50 \
               compute_overlap \
               subpixel_accuracy \
               computation_parameters=[Save computation time (but use more RAM)] \
               image_output=[Fuse and display]")
        fpath = os.path.join(outputdir, '{:04d}.tif'.format(slc))
        IJ.saveAs("Tiff", fpath)


if __name__ == "__main__":
    main(sys.argv[1:])
