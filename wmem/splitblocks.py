#!/usr/bin/env python

"""Convert a directory of tifs to an hdf5 stack.

"""

import sys
import argparse
import os

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

from wmem import parse, utils


def main(argv):
    """"Convert a directory of tifs to an hdf5 stack."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_splitblocks(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    splitblocks(
        args.inputfile,
        args.dset_name,
        args.blocksize,
        args.margin,
        args.usempi & ('mpi4py' in sys.modules),
        args.outputdir,
        args.save_steps,
        args.protective,
        )


def splitblocks(
        h5path_in,
        dset_name,
        blocksize=[500, 500, 500],
        margin=[20, 20, 20],
        usempi=False,
        outputdir='',
        save_steps=False,
        protective=False,
        ):
    """"Convert a directory of tifs to an hdf5 stack."""

    # Prepare for processing with MPI.
    mpi_info = utils.get_mpi_info(usempi)

    # Determine the outputpaths.
    basepath, h5path_dset = h5path_in.split('.h5/')
    datadir, fname = os.path.split(basepath)
    postfix = fname.split(dset_name)[-1]
    if not outputdir:
        blockdir = 'blocks_{:04d}'.format(blocksize[0])
        outputdir = os.path.join(datadir, blockdir)
    utils.mkdir_p(outputdir)
    fname = '{}_{}{}.h5'.format(dset_name, '{}', postfix)
    h5path_tpl = os.path.join(outputdir, fname, h5path_dset)

    # Open data for reading.
    h5_info = utils.h5_load(h5path_in, comm=mpi_info['comm'])
    h5file_in, ds_in, elsize, axlab = h5_info

    # Divide the data into a series of blocks.
    blocks = get_blocks(ds_in.shape, blocksize, margin, h5path_tpl)
    series = np.array(range(0, len(blocks)), dtype=int)
    if mpi_info['enabled']:
        series = utils.scatter_series(mpi_info, series)[0]

    # Write blocks to the outputfile(s).
    for blocknr in series:
        block = blocks[blocknr]
        write_block(ds_in, elsize, axlab, block)

    # Close the h5 files or return the output array.
    try:
        h5file_in.close()
    except (ValueError, AttributeError):
        pass
    except UnboundLocalError:
        pass


def get_blocks(shape, blocksize, margin, h5path_tpl):
    """Create a list of dictionaries with data block info."""

    blockbounds, blocks = {}, []
    for i, dim in enumerate('zyx'):
        blockbounds[dim] = get_blockbounds(shape[i],
                                           blocksize[i],
                                           margin[i])

    for x, X in blockbounds['x']:
        for y, Y in blockbounds['y']:
            for z, Z in blockbounds['z']:
                block = {}
                idstring = '{:05d}-{:05d}_{:05d}-{:05d}_{:05d}-{:05d}'
                block['id'] = idstring.format(x, X, y, Y, z, Z)
                block['slc'] = [slice(z, Z), slice(y, Y), slice(x, X)]
                block['size'] = utils.slices2sizes(block['slc'])
                block['h5path'] = h5path_tpl.format(block['id'])
                blocks.append(block)

    return blocks


def get_blockbounds(shape, blocksize, margin):
    """Get the block range for a dimension."""

    # blocks
    starts = range(0, shape, blocksize)
    stops = np.array(starts) + blocksize

    # blocks with margin
    starts = np.array(starts) - margin
    stops = np.array(stops) + margin

    # blocks with margin reduced on boundary blocks
    starts[starts < 0] = 0
    stops[stops > shape] = shape

    return zip(starts, stops)


def write_block(ds_in, elsize, axlab, block):
    """Write the block to file."""

    shape = list(block['size'])
    if ds_in.ndim == 4:
        shape += [ds_in.shape[3]]

    chunks = ds_in.chunks
    if any(np.array(chunks) > np.array(shape)):
        chunks = True

    h5file_out, ds_out = utils.h5_write(
        data=None,
        shape=shape,
        dtype=ds_in.dtype,
        h5path_full=block['h5path'],
        chunks=chunks,
        element_size_um=elsize,
        axislabels=axlab,
        )

    slcs = block['slc']
    if ds_in.ndim == 3:
        ds_out[:] = ds_in[slcs[0], slcs[1], slcs[2]]
    elif ds_in.ndim == 4:
        ds_out[:] = ds_in[slcs[0], slcs[1], slcs[2], :]

    h5file_out.close()


if __name__ == "__main__":
    main(sys.argv)
