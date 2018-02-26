#!/usr/bin/env python

"""Combine volumes by addition.

"""

import sys
import argparse

import numpy as np

from wmem import parse, utils


def main(argv):
    """Combine volumes by addition."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_combine_vols(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    combine_vols(
        args.inputfile,
        args.volidxs,
        args.outputfile,
        args.protective,
        )


def combine_vols(
        h5path_in,
        vol_idxs=[0, 2, 4, 7],
        h5path_out='',
        protective=False,
        ):
    """Combine volumes by addition."""

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
                                        element_size_um=es[:3],
                                        axislabels=al[:3])

    out = np.zeros(ds_out.shape, dtype=ds_out.dtype)
    for volnr in vol_idxs:
        out += ds_in[:, :, :, volnr]
    ds_out[:] = out

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


if __name__ == "__main__":
    main(sys.argv[1:])
