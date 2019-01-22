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
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.volidxs,
        args.outputfile,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def combine_vols(
        image_in,
        dataslices=None,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        vol_idxs=[],
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Combine volumes by addition."""

    mpi_info = utils.get_mpi_info(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, comm=mpi_info['comm'],
                         dataslices=dataslices)

    # Open the outputfile for writing and create the dataset or output array.
    props = im.get_props(protective=protective, squeeze=True)
    mo = Image(outputpath, **props)
    mo.create(comm=mpi_info['comm'])
    in2out_offset = -np.array([slc.start for slc in mo.slices])

    # Prepare for processing with MPI.
    blocks = utils.get_blocks(im, blocksize, blockmargin, blockrange)
    series = utils.scatter_series(mpi_info, len(blocks))[0]

    for blocknr in series:

        out = add_volumes(im, blocks[blocknr], vol_idxs)

        slcs_out = squeeze_slices(im.slices, im.axlab.index('c'))
        slcs_out = im.get_offset_slices(in2out_offset)
        mo.write(data=out, slices=slcs_out)

    im.close()
    mo.close()

    return mo


def add_volumes(im, block, vol_idxs):
    """"""

    c_axis = im.axlab.index('c')
    size = get_outsize(im, block['slices'])
    out = np.zeros(size, dtype=im.dtype)

    im.slices = block['slices']
    if vol_idxs:
        for volnr in vol_idxs:
            im.slices[c_axis] = slice(volnr, volnr + 1, 1)
            out += im.slice_dataset()
    else:
        out = np.sum(im.slice_dataset(), axis=c_axis)

    return out


def squeeze_slices(slices, axis):

    slcs = list(slices)
    del slcs[axis]

    return slcs


def get_outsize(im, slices):

    slcs = squeeze_slices(slices, im.axlab.index('c'))
    size = list(im.slices2shape(slcs))

    return size


if __name__ == "__main__":
    main(sys.argv[1:])
