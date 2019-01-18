#!/usr/bin/env python

"""Convert a directory of tifs to an hdf5 stack.

"""

import sys
import argparse
import os

import numpy as np

import mpi4py
try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

from wmem import parse, utils, Image


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
        args.inputpath,
        args.dataslices,
        args.dset_name,
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.outputpath,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def splitblocks(
        image_in,
        dataslices=None,
        dset_name='',
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """"Convert a directory of tifs to an hdf5 stack."""

    # Prepare for processing with MPI.
    mpi_info = utils.get_mpi_info(usempi)

    # Open data for reading.
    im = utils.get_image(image_in, comm=mpi_info['comm'],
                         dataslices=dataslices)

    # Divide the data into a series of blocks.
    tpl = get_template_string(im, blocksize, dset_name, outputpath)
    blocks = utils.get_blocks(im, blocksize, blockmargin, blockrange, tpl)
    series = utils.scatter_series(mpi_info, len(blocks))[0]

    # Write blocks to the outputfile(s).
    for blocknr in series:
        write_block(im, blocks[blocknr], protective, comm=mpi_info['comm'])

    im.close()


def write_block(im, block, protective=False, comm=None):
    """Write the block to file."""

    im.dataslices = block['dataslices']
    im.load()
    data = im.slice_dataset()

    shape = list(block['size'])
    if im.get_ndim() == 4:
        shape += [im.ds.shape[3]]

    chunks = im.chunks
    if any(np.array(chunks) > np.array(shape)):
        chunks = True

    mo = Image(block['path'],
               elsize=im.elsize,
               axlab=im.axlab,
               shape=shape,
               chunks=im.chunks,
               dtype=im.dtype,
               protective=protective)
    mo.create(comm)
    mo.write(data=data)
    mo.close()


def get_template_string(im, blocksize, dset_name='', outputpath=''):  # FIXME

    comps = im.split_path(outputpath)

    if '{}' in outputpath:  # template provided
        template_string = outputpath
        utils.mkdir_p(comps['dir'])
        return template_string

    if not outputpath:
        blockdir = 'blocks_{:04d}'.format(blocksize[0])
        outputdir = os.path.join(comps['dir'], blockdir)

    utils.mkdir_p(outputdir)

    if im.format == '.h5':
        if dset_name:
            postfix = comps['fname'].split(dset_name)[-1]
            fname = '{}_{}{}{}'.format(dset_name, '{}', postfix, comps['ext'])
        else:
            fname = '{}_{}{}'.format(comps['fname'], '{}', comps['ext'])

        template_string = os.path.join(outputdir, fname, comps['dset'])
    else:
        fname = '{}_{}{}'.format(comps['fname'], '{}', comps['ext'])
        template_string = os.path.join(outputdir, fname)

    return template_string


if __name__ == "__main__":
    main(sys.argv)
