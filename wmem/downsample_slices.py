#!/usr/bin/env python

"""Downsample a series of 2D images.

Downsampled images are written to the output directory
with the same filename as the input images.
- A subset of the data can be selected with python slicing:
providing <start stop step> in (plane, row, col) order
e.g. setting the flag -D <20 44 1 100 200 0 0 1>
selects the images 20 through 43 yielded by the regular expression
and makes a cutout of rows 100 through 199
and with the columns set to the full dimension.
- MPI can be enabled for this function.
- No intermediate results are saved.
- Protective mode will check if images with the same name
already exist in the output directory.
"""

import sys
import argparse
import os
import glob

from scipy.misc import imsave
import numpy as np
from skimage import io
from skimage.transform import resize

from wmem import parse, utils


def main(argv):
    """Downsample a series of 2D images."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_downsample_slices(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    downsample_slices(
        args.inputdir,
        args.outputdir,
        args.regex,
        args.downsample_factor,
        args.dataslices,
        args.usempi & ('mpi4py' in sys.modules),
        args.protective,
        )


def downsample_slices(
        inputdir,
        outputdir,
        regex='*.tif',
        ds_factor=4,
        dataslices=None,
        use_mpi=False,
        protective=False,
        ):
    """Downsample a series of 2D images."""

    # Get the list of input filepaths.
    files = sorted(glob.glob(os.path.join(inputdir, regex)))

    # Get the list of output filepaths.
    utils.mkdir_p(outputdir)
    outpaths = []
    for fpath in files:
        root, ext = os.path.splitext(fpath)
        tail = os.path.split(root)[1]
        outpaths.append(os.path.join(outputdir, tail + ext))
    # Check if any output paths already exist.
    status = utils.output_check_dir(outpaths, protective)
    if status == "CANCELLED":
        return

    # Get the slice objects for the input data.
    zyxdims = [len(files)] + imread(files[0]).shape
    slices = utils.get_slice_objects_prc(dataslices, zyxdims)

    # Prepare for processing with MPI.
    series = np.array(range(slices[0].start,
                            slices[0].stop,
                            slices[0].step), dtype=int)
    if use_mpi:
        mpi_info = utils.get_mpi_info()
        series = utils.scatter_series(mpi_info, series)[0]

    # Downsample and save the images.
    for slc in series:
        sub = io.imread(files[slc])[slices[1], slices[2]]
        downsample_image(outpaths[slc], sub, ds_factor)


def downsample_image(outputpath, img, ds_factor=4):
    """Downsample and save the image."""

    img_ds = resize(img, (img.shape[0] / ds_factor,
                          img.shape[1] / ds_factor))
    imsave(outputpath, img_ds)


if __name__ == "__main__":
    main(sys.argv[1:])
