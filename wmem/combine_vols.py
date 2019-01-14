#!/usr/bin/env python

"""Combine volumes by addition.

"""

import sys
import argparse

import numpy as np

from wmem import parse, utils, Image


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
        args.dataslices,
        args.volidxs,
        args.outputfile,
        args.protective,
        )


def combine_vols(
        image_in,
        dataslices=None,
        vol_idxs=[],
        outputpath='',
        protective=False,
        ):
    """Combine volumes by addition."""

    # Open data for reading.
    im = utils.get_image(image_in, dataslices=dataslices)
    c_axis = im.axlab.index('c')
    squeezed = im.squeeze_channel(c_axis)
    dims = im.slices2shape()  # FIXME

    # Open the outputfile for writing and create the dataset or output array.
    mo = Image(outputpath,
               shape=dims,
               dtype=im.dtype,
               elsize=squeezed['es'],
               axlab=squeezed['al'],
               chunks=squeezed['chunks'],
               dataslices=squeezed['dslcs'],
               protective=protective)
    mo.create()

    out = np.zeros(mo.dims, dtype=mo.dtype)
    if vol_idxs:
        for volnr in vol_idxs:
            im.dataslices[c_axis*3] = volnr
            im.dataslices[c_axis*3+1] = volnr + 1
            im.dataslices[c_axis*3+2] = 1
            out += im.slice_dataset()
    else:
        out = np.sum(im.slice_dataset(), axis=c_axis)

    mo.write(data=out)

    im.close()
    mo.close()

    return mo


if __name__ == "__main__":
    main(sys.argv[1:])
