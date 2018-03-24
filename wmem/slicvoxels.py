#!/usr/bin/env python

"""Calculate SLIC supervoxels.

"""

import os
import sys
import argparse

import numpy as np
from skimage import segmentation

from wmem import parse, utils


def main(argv):
    """Calculate SLIC supervoxels."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_slicvoxels(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    slicvoxels(
        args.inputfile,
        args.slicvoxelsize,
        args.compactness,
        args.sigma,
        args.enforce_connectivity,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def slicvoxels(
        h5path_in,
        slicvoxelsize=500,
        compactness=0.2,
        sigma=1,
        enforce_connectivity=False,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Calculate SLIC supervoxels."""

    # check output paths
    outpaths = {'out': h5path_out}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape[:3], 'uint64',
                                        h5path_out,
                                        element_size_um=elsize[:3],
                                        axislabels=axlab[:3])

    n_segments = int(ds_in.size / slicvoxelsize)
    spac = [es for es in np.absolute(elsize[:3])]  # TODO: set spac for elsize=None

    data = utils.normalize_data(ds_in[:])[0]
    segments = segmentation.slic(data,
                                 n_segments=n_segments,
                                 compactness=compactness,
                                 spacing=spac,
                                 sigma=sigma,
                                 multichannel=ds_in.ndim == 4,
                                 convert2lab=False,
                                 enforce_connectivity=enforce_connectivity)

    ds_out[:] = segments + 1
    print("Number of supervoxels: ", np.amax(segments) + 1)

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


if __name__ == "__main__":
    main(sys.argv[1:])

