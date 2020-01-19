#!/usr/bin/env python

"""Combine volumes by addition.

"""

import sys
import argparse

import numpy as np

from skimage.transform import resize

from wmem import parse, utils, Image, wmeMPI, done, get_image
from wmem.splitblocks import get_template_string

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
        args.bias_image,
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
        bias_image='',
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Combine volumes by addition."""

    write_to_single_file = False  # TODO?: make argument option

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    im = Image(image_in, dataslices=dataslices, permission='r')
    im.load(comm=mpi.comm, load_data=False)

    if bias_image:
        bf = Image(bias_image, permission='r')
        bf.load(load_data=False)
    else:
        bf = None

    # Prepare for processing with MPI.
    if not write_to_single_file:
        tpl = get_template_string(im, blocksize=blocksize, dset_name='', outputpath=outputpath)
    else:
        tpl = ''
    mpi.set_blocks(im, blocksize, blockmargin, blockrange, tpl)
    mpi.scatter_series()

    # Open the outputfile for writing and create the dataset or output array.
    props = im.get_props(protective=protective)
    if len(props['shape']) == 5:
        props = im.squeeze_props(props=props, dim=4)
    if len(props['shape']) == 4:
        props = im.squeeze_props(props=props, dim=3)
#     props['shape'] = get_outsize(im, mpi.blocks[0]['slices'])
    if write_to_single_file:
        mo = Image(outputpath, **props)
        mo.create(comm=mpi.comm)
        in2out_offset = -np.array([slc.start for slc in mo.slices])

    for i in mpi.series:
        block = mpi.blocks[i]
        print('Processing blocknr {:4d} with id: {}'.format(i, block['id']))

        out = add_volumes(im, block, vol_idxs, bf)
        mean = True
        if mean:
            out = out / len(vol_idxs)

        if write_to_single_file:
            slcs_out = squeeze_slices(im.slices, im.axlab.index('c'))
            slcs_out = im.get_offset_slices(in2out_offset)
            mo.write(data=out, slices=slcs_out)
        else:
            props['shape'] = out.shape
            mo = Image(block['path'], **props)
            mo.create(comm=mpi.comm)
            mo.slices = None
            mo.set_slices()
            mo.write(data=out)
            mo.close()

    im.close()
    if write_to_single_file:
        mo.close()

    done()
    return mo


def add_volumes(im, block, vol_idxs, bf=None):
    """"""

    c_axis = im.axlab.index('c')
    size = get_outsize(im, block['slices'])
#     size = list(im.slices2shape(block['slices']))
    out = np.zeros(size, dtype='float64')

    im.slices = block['slices']
    if vol_idxs:
        for volnr in vol_idxs:
            im.slices[c_axis] = slice(volnr, volnr + 1, 1)
            data = im.slice_dataset()
            if bf is not None:
                bias = get_bias_field_block(bf, im.slices)
                bias = np.reshape(bias, data.shape)
                data /= bias
                data = np.nan_to_num(data, copy=False)
            out += data
    else:
        out = np.sum(im.slice_dataset(), axis=c_axis)

    return out


def get_bias_field_block(bf, slices, dsfacs=[1, 64, 64, 1]):

    bf.slices = [slice(int(slc.start / ds), int(slc.stop / ds), 1)
                 for slc, ds in zip(slices, dsfacs)]
    bf_block = bf.slice_dataset().astype('float32')
    outdims = list(bf.slices2shape(slices))
    bias = resize(bf_block, outdims, preserve_range=True)

    return bias


def squeeze_slices(slices, axis):

    slcs = list(slices)
    del slcs[axis]

    return slcs


def get_outsize(im, slices):

    slcs = list(slices)
    if 't' in im.axlab:
        slcs = squeeze_slices(slices, im.axlab.index('t'))
    if 'c' in im.axlab:
        slcs = squeeze_slices(slcs, im.axlab.index('c'))
    size = list(im.slices2shape(slcs))

    return size


if __name__ == "__main__":
    main(sys.argv[1:])
