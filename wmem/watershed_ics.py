#!/usr/bin/env python

"""Perform watershed on the intracellular space compartments.

"""

import os
import sys
import argparse

import numpy as np
from scipy.ndimage import label
from skimage.morphology import watershed, remove_small_objects
from skimage.segmentation import relabel_sequential

from wmem import parse, utils, wmeMPI, Image, LabelImage, MaskImage


def main(argv):
    """Perform watershed on the intracellular space compartments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_watershed_ics(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    watershed_ics(
        args.inputfile,
        args.dataslices,
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.mask_in,
        args.seedimage,
        args.seed_size,
        args.lower_threshold,
        args.upper_threshold,
        args.invert,
        args.outputfile,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def watershed_ics(
        image_in,
        dataslices=None,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        mask_in='',
        seeds_in='',
        seed_size=64,
        lower_threshold=0.00,
        upper_threshold=1.00,
        invert=False,
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Perform watershed on the intracellular space compartments."""

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices)

    # Open the outputfiles for writing and create the dataset or output array.
    props = im.get_props(protective=protective, dtype='uint64', squeeze=True)
    outpaths = get_outpaths(outputpath, save_steps)
    mo = LabelImage(outpaths['out'], **props)
    mo.create(comm=mpi.comm)  # FIXME: load if exist
    in2out_offset = -np.array([slc.start for slc in mo.slices])

    # FIXME: if seeds h5 already exists with different dims it will load and fail
    # load/generate the seeds and mask
    seeds = get_seeds(seeds_in, mpi, outpaths['seeds'], **props)
    mask = get_mask(mask_in, mpi, outpaths['mask'], **props)

    # Prepare for processing with MPI.
    mpi.set_blocks(im, blocksize, blockmargin, blockrange)
    mpi.scatter_series()

    for i in mpi.series:
        block = mpi.blocks[i]

        for img in [im, mask, seeds]:
            img.slices = block['slices']

        slices_out = im.get_offset_slices(in2out_offset)

        data = im.slice_dataset()
        if mask_in:
            maskdata = mask.slice_dataset().astype('bool')
        else:
            maskdata = np.ones(im.dims, dtype='bool')
            mask.write(data=maskdata, slices=mask.slices)

        if seeds_in:
            seedsdata = seeds.slice_dataset()
        else:
            seedsdata = calculate_seeds(data,
                                        lower_threshold,
                                        upper_threshold,
                                        seed_size)
            seeds.write(seedsdata, slices_out)

        # perform the watershed
        if invert:
            ws = watershed(-data, seedsdata, mask=maskdata)
        else:
            ws = watershed(data, seedsdata, mask=maskdata)

        mo.write(data=ws, slices=slices_out)

    im.close()
    mo.close()
    seeds.close()
    mask.close()

    return mo


def get_outpaths(h5path_out, save_steps):

    outpaths = {'out': h5path_out, 'seeds': '', 'mask': ''}
    if save_steps:
        outpaths = utils.gen_steps(outpaths, save_steps)

    return outpaths


def get_seeds(seeds_in, mpi, outpath='', **kwargs):

    kwargs['dtype'] = 'uint64'

    if seeds_in:
        seeds = utils.get_image(seeds_in, comm=mpi.comm,
                                slices=kwargs['slices'])
    else:
        seeds = LabelImage(outpath, **kwargs)
        seeds.create(comm=mpi.comm)

    return seeds


def calculate_seeds(data, lower_threshold, upper_threshold, min_seed_size):
    """Extracts watershed seeds from data."""

    lower_threshold = lower_threshold or np.amin(data) - 1
    upper_threshold = upper_threshold or np.amax(data) + 1
    seeds = np.logical_and(data > lower_threshold, data <= upper_threshold)
    seeds = label(seeds)[0]
    seeds = remove_small_objects(seeds, min_size=min_seed_size)
    seeds = relabel_sequential(seeds)[0]
    """ NOTE:
    numpy/scipy/skimage inplace is not written when using hdf5
    Therefore, we cannot use:
    np.logical_and(ds_in[:] > lower_threshold,
                   ds_in[:] <= upper_threshold, ds_sds[:])
    num = label(ds_sds[:], output=ds_sds[:])
    remove_small_objects(ds_sds[:], min_size=seed_size, in_place=True)
    """

    return seeds


def get_mask(mask_in, mpi, outpath='', **kwargs):

    kwargs['dtype'] = 'bool'

    if mask_in:
        mask = utils.get_image(mask_in, comm=mpi.comm, slices=kwargs['slices'])
    else:
        mask = MaskImage(outpath, **kwargs)
        mask.create(comm=mpi.comm)

    return mask


def calculate_mask(maskimage, masks):
    """Extracts watershed seeds from data."""

    dims = list(maskimage.slices2shape())
    maskdata = np.ones(dims, dtype='bool')
    if masks:
        dataslices = utils.slices2dataslices(maskimage.slices)
        maskdata = utils.string_masks(masks, maskdata, dataslices)

    maskimage.write(data=maskdata, slices=maskimage.slices)

    return maskdata


if __name__ == "__main__":
    main(sys.argv[1:])
