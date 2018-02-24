#!/usr/bin/env python

"""Create thresholded hard segmentations.

"""

import sys
import argparse

import numpy as np

from wmem import parse, utils


def main(argv):
    """Create thresholded hard segmentations."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_combine_vols(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    combine_vols(
        args.inputfile,
        args.outputfile,
        args.protective,
        )


def combine_vols(
        h5path_in,
        h5path_out='',
        protective=False,
        ):
    """Create thresholded hard segmentation."""

    if bool(h5path_out) and ('.h5' in h5path_out):
        status, info = utils.h5_check(h5path_out, protective)
        print(info)
        if status == "CANCELLED":
            return

    # open data for reading
    h5file_in, ds_in, es, al = utils.load(h5path_in)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape[:3], ds_in.dtype,
                                        h5path_out,
                                        element_size_um=es,
                                        axislabels=al)

    ds_out[:] = ds_in[:, :, :, 0] + ds_in[:, :, :, 2] + ds_in[:, :, :, 4] + ds_in[:, :, :, 7]

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


if __name__ == "__main__":
    main(sys.argv[1:])
