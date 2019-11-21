#!/usr/bin/env python

""".

"""

import os
import sys
import argparse

from scipy.ndimage.filters import gaussian_filter

from wmem import parse, utils, wmeMPI, Image


def main(argv):
    """."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_image_ops(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    image_ops(
        args.inputfile,
        args.dataslices,
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.sigma,
        args.outputfile,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def image_ops(
        image_in,
        dataslices=None,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        sigma=[0.0],
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """."""

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices)
    props = im.get_props(protective=protective)

    # Open the outputfiles for writing and create the dataset or output array.
    mo = Image(outputpath, **props)
    mo.create(comm=mpi.comm)

    # Prepare for processing with MPI.
    mpi.set_blocks(im, blocksize, blockmargin, blockrange)
    mpi.scatter_series()

    for i in mpi.series:
        block = mpi.blocks[i]

        im.slices = mo.slices = block['slices']
        data = im.slice_dataset()

        if any(sigma):
            data = smooth(data, sigma, im.elsize)

        mo.write(data)

    im.close()
    mo.close()

    return mo


def smooth(data, sigma, elsize):

    if len(sigma) == 1:
        sigma = sigma * len(elsize)
    elif len(sigma) != len(elsize):
        raise Exception('sigma does not match dimensions')
    sigma = [sig / es for sig, es in zip(sigma, elsize)]

    data_smoothed = gaussian_filter(data, sigma)

    return data_smoothed


if __name__ == "__main__":
    main(sys.argv[1:])
