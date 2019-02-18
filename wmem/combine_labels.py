#!/usr/bin/env python

""".

"""

import os
import sys
import argparse

import numpy as np

from wmem import parse, utils, wmeMPI, LabelImage


def main(argv):
    """."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_combine_labels(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    combine_labels(
        args.inputfile1,
        args.inputfile2,
        args.dataslices,
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.method,
        args.outputfile,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def combine_labels(
        image1_in,
        image2_in,
        dataslices=None,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        method='add',
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """."""

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    im1 = utils.get_image(image1_in, comm=mpi.comm, dataslices=dataslices)
    im2 = utils.get_image(image2_in, comm=mpi.comm, dataslices=dataslices)
    props = im1.get_props(protective=protective)

    # Open the outputfiles for writing and create the dataset or output array.
    mo = LabelImage(outputpath, **props)
    mo.create(comm=mpi.comm)

    # Prepare for processing with MPI.
    mpi.set_blocks(im1, blocksize, blockmargin, blockrange)
    mpi.scatter_series()

    for i in mpi.series:
        block = mpi.blocks[i]

        im1.slices = im2.slices = mo.slices = block['slices']
        labels1 = im1.slice_dataset()
        labels2 = im2.slice_dataset()

        if method == 'add':
            labels = labels1 + labels2
        elif method == 'subtract':
            labels = labels1 - labels2
        elif method == 'mask':
            mask = labels2.astype('bool')
            labels = np.copy(labels1)
            labels[mask] = 0

        mo.write(labels)

    im1.close()
    im2.close()
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
