#!/usr/bin/env python

"""Register the slices in a stack.

ImageJ headless doesn't take arguments easily...
Example usage on Oxford's ARC arcus-b cluster:

scriptdir="${HOME}/workspace/EM"
imagej=/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64
datadir="${DATA}/EM/M3/M3_S1_GNU"
refname="0000.tif"

mkdir -p $datadir/reg/trans

sed "s?SOURCE_DIR?$datadir/stitched?;\
    s?TARGET_DIR?$datadir/reg?;\
    s?REFNAME?$refname?;\
    s?TRANSF_DIR?$datadir/reg/trans?g" \
    $scriptdir/reg/fiji_register.py \
    > $datadir/fiji_register.py

qsubfile=$datadir/fiji_register_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_reg" >> $qsubfile
echo "$imagej --headless \\" >> $qsubfile
echo "$datadir/fiji_register.py" >> $qsubfile
sbatch $qsubfile
"""

import sys

from ij import IJ
from register_virtual_stack import Register_Virtual_Stack_MT


def main(argv):
    """Register the slices in a stack."""

    source_dir = 'SOURCE_DIR/'
    target_dir = 'TARGET_DIR/'
    transf_dir = 'TARGET_DIR/trans/'
    reference_name = 'REFNAME'

    use_shrinking_constraint = 0
    p = Register_Virtual_Stack_MT.Param()
    p.sift.maxOctaveSize = 1024
    p.minInlierRatio = 0.05
    Register_Virtual_Stack_MT.exec(source_dir, target_dir, transf_dir,
                                   reference_name, p, use_shrinking_constraint)


if __name__ == "__main__":
    main(sys.argv)
