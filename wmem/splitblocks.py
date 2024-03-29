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

from wmem import parse, utils, wmeMPI, Image, done
from wmem.stack2stack import remove_singleton, permute_axes


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
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.dset_name,
        args.outlayout,
        args.chunksize,
        args.outputpath,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def splitblocks(
        image_in,
        dataslices=None,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        dset_name='',
        outlayout=None,
        chunksize=[],
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """"Convert a directory of tifs to an hdf5 stack."""

    mpi = wmeMPI(usempi)

    # Open data for reading.
    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices, load_data=False)

    # Prepare for processing with MPI.
    tpl = get_template_string(im, blocksize, dset_name, outputpath)
    mpi.set_blocks(im, blocksize, blockmargin, blockrange, tpl)
    mpi.scatter_series()

    # Write blocks to the outputfile(s).
    for i in mpi.series:
        block = mpi.blocks[i]
        print('Processing blocknr {:4d} with id: {}'.format(i, block['id']))

        props = {}
        props['axlab'] = outlayout or str(im.axlab)
        props['elsize'] = list(im.elsize)
        if chunksize is not None:
            props['chunks'] = list(chunksize)
        else: 
            props['chunks'] = list(im.chunks)

        write_block(im, block, props, comm=mpi.comm)

    im.close()
    done()


def write_block(im, block, props, comm=None):
    """Write the block to file."""

    im.slices = block['slices']
    data = im.slice_dataset(squeeze=False)

    if im.format == '.ims':
        axlab = 'zyxct'
    else:
        axlab = str(im.axlab)

    in2out_data = [axlab.index(l) for l in props['axlab']]
    data = np.transpose(data, in2out_data)
    props['shape'] = list(data.shape)

    in2out_props = [im.axlab.index(l) for l in props['axlab']]
    for prop in ['elsize', 'chunks']:
        if props[prop] is not None:
            props[prop] = [props[prop][i] for i in in2out_props]

    props, data = remove_singleton(im, props, data, props['axlab'])

    for i, (ch, sh) in enumerate(zip(props['chunks'], props['shape'])):
        if ch > sh:
            props['chunks'][i] = sh

    props['dtype'] = data.dtype
    mo = Image(block['path'], **props)
    mo.create(comm)
    mo.slices = None
    mo.set_slices()
    mo.write(data=data)
    mo.close()


def get_template_string(im, blocksize, dset_name='', outputpath=''):  # FIXME

    comps = im.split_path(outputpath, fileformat=im.get_format(outputpath))

    outputdir = comps['dir']
    if not outputpath:
        blockdir = 'blocks_{:04d}'.format(blocksize[2])
        outputdir = os.path.join(outputdir, blockdir)

    utils.mkdir_p(outputdir)

    if '{}' in outputpath:  # template provided
        return outputpath

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
