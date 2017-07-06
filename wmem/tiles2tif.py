#!/usr/bin/env python

"""Convert .dm3 files to tifs.

ImageJ headless doesn't take arguments easily...
Example usage on Oxford's ARC arcus cluster:

scriptdir="${HOME}/workspace/EM"
dm3dir="${DATA}/EM/M3/20Mar15/montage/Montage_"
datadir="${DATA}/EM/M3/M3_S1_GNU"
imagej=/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64

mkdir -p $datadir/tifs

for montage in 000 001 002 003; do
sed "s?INPUTDIR?$dm3dir$montage?;\
    s?OUTPUTDIR?$datadir/tifs?;\
    s?OUTPUT_POSTFIX?_m$montage?g" \
    $scriptdir/convert/tiles2tif.py \
    > $datadir/EM_tiles2tif_m$montage.py
done

qsubfile=$datadir/EM_tiles2tif_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=01:00:00" >> $qsubfile
echo "#PBS -N em_tif" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "$imagej --headless \\" >> $qsubfile
echo "$datadir/EM_tiles2tif_m\`printf %03d \$PBS_ARRAYID\`.py" >> $qsubfile
qsub -t 0-3 $qsubfile
"""

import sys
import os
import glob

from loci.plugins import BF
from ij import IJ


def main(argv):
    """Convert .dm3 files to tifs."""

    inputdir = 'INPUTDIR'
    outputdir = 'OUTPUTDIR'
    output_postfix = 'OUTPUT_POSTFIX'

    infiles = glob.glob(os.path.join(inputdir, '*.dm3'))

    for infile in infiles:
        imp = BF.openImagePlus(infile)
        tail = os.path.split(infile)[1]
        filename = os.path.splitext(tail)[0][-4:] + output_postfix + '.tif'
        outpath = os.path.join(outputdir, filename)
        IJ.save(imp[0], outpath)


if __name__ == "__main__":
    main(sys.argv[1:])
