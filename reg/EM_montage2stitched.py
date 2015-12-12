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
        IJ.saveAs("Tiff", path.join(outputdir, str(slc).zfill(4) + ".tif"))

if __name__ == "__main__":
    main(sys.argv[1:])

