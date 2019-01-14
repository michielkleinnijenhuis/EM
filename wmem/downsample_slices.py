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
        usempi=False,
        protective=False,
        ):
    """Downsample a series of 2D images."""

    if '.h5' in outputdir:
        status, info = utils.h5_check(outputdir, protective)
        print(info)
        if status == "CANCELLED":
            return

    if '.h5' in inputdir:  # FIXME: assumed zyx for now

        h5file_in, ds_in, elsize, axlab = utils.h5_load(inputdir)
        zyxdims = ds_in.shape

    else:

        # Get the list of input filepaths.
        files = sorted(glob.glob(os.path.join(inputdir, regex)))
        zyxdims = [len(files)] + list(io.imread(files[0]).shape)
        axlab = 'zyx'

    if '.h5' in outputdir:

        elsize[1] = elsize[1] / ds_factor
        elsize[2] = elsize[2] / ds_factor
        outsize = [ds_in.shape[0],
                   ds_in.shape[1] / ds_factor,
                   ds_in.shape[2] / ds_factor]
        h5file_out, ds_out = utils.h5_write(None, outsize, ds_in.dtype,
                                            outputdir,
                                            element_size_um=elsize,
                                            axislabels=axlab)

    else:

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
    slices = utils.get_slice_objects_prc(dataslices, zyxdims)
    # Prepare for processing with MPI.
    mpi_info = utils.get_mpi_info(usempi)
    series = np.array(range(slices[0].start,
                            slices[0].stop,
                            slices[0].step), dtype=int)
    if mpi_info['enabled']:
        series = utils.scatter_series(mpi_info, len(series))[0]

    # Downsample and save the images.
    for slc in series:
        if '.h5' in inputdir:
            sub = ds_in[slc, slices[1], slices[2]]
        else:
            sub = io.imread(files[slc])[slices[1], slices[2]]

        img_ds = resize(sub, (sub.shape[0] / ds_factor,
                              sub.shape[1] / ds_factor))

        if '.h5' in outputdir:
            ds_out[slc, :, :] = img_ds
        else:
            imsave(outpaths[slc], img_ds)
#         downsample_image(outpaths[slc], sub, ds_factor)

    try:
        h5file_in.close()
        h5file_out.close()
    except (ValueError, AttributeError):
        pass
#         return ds_out


def downsample_image(outputpath, img, ds_factor=4):
    """Downsample and save the image."""

    img_ds = resize(img, (img.shape[0] / ds_factor,
                          img.shape[1] / ds_factor))
    imsave(outputpath, img_ds)


if __name__ == "__main__":
    main(sys.argv[1:])
