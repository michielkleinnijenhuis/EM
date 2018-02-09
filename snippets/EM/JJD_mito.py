#!/usr/bin/env python

import sys
import os
from loci.plugins import BF
from ij import IJ


def main(argv):
    """Segment mitochondria."""

    datadir = '/Users/michielk/M3S1GNUds7'
    filename = 'M3S1GNUds7_tiff.tif'
    filepath = os.path.join(datadir, filename)
    imp = BF.openImagePlus(filepath)

#     for infile in infiles:
#         imp = BF.openImagePlus(infile)
#         head, tail = path.split(infile)
#         filename, ext = path.splitext(tail)
#         outpath = path.join(outputdir, filename[-4:] + output_postfix + '.tif')
#         IJ.save(imp[0], outpath)


if __name__ == "__main__":
    main(sys.argv[1:])
