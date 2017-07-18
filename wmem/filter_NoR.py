#!/usr/bin/env python

"""Filter nodes of ranvier.

"""

import sys
import argparse

import numpy as np
from skimage.measure import regionprops
try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")

from wmem import parse, utils


# TODO: write elsize and axislabels
def main(argv):
    """Filter nodes of ranvier."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_filter_NoR(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    filter_NoR(
        args.inputfile,
        args.input2D,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def filter_NoR(
        h5path_in,
        h5path_2D,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Filter nodes of ranvier."""

    # check output paths
    outpaths = {'out': h5path_out}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
    h5file_2D, ds_2D, _, _ = utils.h5_load(h5path_2D)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(ds_in, ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    labelsets = {i: set(np.unique(ds_2D[i, :, :]))
                 for i in range(ds_2D.shape[0])}

    ulabels = np.unique(ds_in)
    m = {l: np.array([True if l in lsv else False
                      for _, lsv in labelsets.items()])
         for l in ulabels}

    rp = regionprops(ds_in)
    for prop in rp:
        z, y, x, Z, Y, X = tuple(prop.bbox)
        mask = prop.image
        mask[m[prop.label][z:Z], :, :] = 0
        ds_out[z:Z, y:Y, x:X][mask] = 0

    # close and return
    h5file_in.close()
    h5file_2D.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


if __name__ == "__main__":
    main(sys.argv[1:])
