#!/usr/bin/env python

"""Create thresholded hard segmentations.

"""

import sys
import argparse

import numpy as np

from skimage.morphology import remove_small_objects
from skimage.morphology import binary_dilation, ball, disk

from wmem import parse, utils, wmeMPI, Image, MaskImage


def main(argv):
    """Create thresholded hard segmentations."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_prob2mask(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    if args.step:
        for thr in frange(args.step, 1.0, args.step):
            prob2mask(
                args.inputfile,
                args.dataslices,
                thr,
                args.upper_threshold,
                args.size,
                args.dilation,
                args.go2D,
                args.usempi & ('mpi4py' in sys.modules),
                args.outputfile.format(thr),
                args.save_steps,
                args.protective,
                )
    else:
        prob2mask(
            args.inputpath,
            args.dataslices,
            args.lower_threshold,
            args.upper_threshold,
            args.size,
            args.dilation,
            args.go2D,
            args.outputpath,
            args.save_steps,
            args.protective,
            args.usempi & ('mpi4py' in sys.modules),
            )


def prob2mask(
        image_in,
        dataslices=None,
        lower_threshold=0,
        upper_threshold=1,
        size=0,
        dilation=0,
        go2D=False,
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Create thresholded hard segmentation."""

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices)

    # Open the outputfile for writing and create the dataset or output array.
    props = im.get_props(protective=protective, dtype='bool', squeeze=True)
    outpaths = get_outpaths(outputpath, save_steps, dilation, size)
    mos = {}
    for stepname, outpath in outpaths.items():
        mos[stepname] = MaskImage(outpath, **props)
        mos[stepname].create(comm=mpi.comm)  # FIXME: load if exist
    in2out_offset = -np.array([slc.start for slc in mos['out'].slices])

    # Prepare for processing with MPI.
    blocksize = [dim for dim in im.dims]
    if go2D:
        blocksize[im.axlab.index('z')] = 1
        se = disk
    else:
        se = ball
    mpi.set_blocks(im, blocksize)
    mpi.scatter_series()

    # Threshold (and dilate and filter) the data.
    for i in mpi.series:
        block = mpi.blocks[i]

        im.slices = block['slices']
        im.load()

        data = im.slice_dataset()
        masks = process_slice(data,
                              lower_threshold, upper_threshold,
                              size, dilation, se, outpaths)

        # FIXME: cannot write nifti/3Dtif in parts
        for stepname, mask in masks.items():
            slcs_out = im.get_offset_slices(in2out_offset)
            mos[stepname].write(data=mask, slices=slcs_out)

    im.close()
    for _, mo in mos.items():
        mo.close()

    return mos['out']


def process_slice(data, lt, ut, size=0, dilation=0,
                  se=disk, outpaths=''):
    """Threshold and filter data."""

    masks = {}
    mask_raw, mask_mito, mask_dil = None, None, None

    if lt or ut:
        mask = np.logical_and(data > lt, data <= ut)
    else:
        mask = data
    mask_raw = mask.copy()
    if 'raw' in outpaths.keys():
        masks['raw'] = mask_raw

    if dilation:
        mask = dilate_mask(mask, dilation, se)
        mask_dil = mask.copy()
        if 'dil' in outpaths.keys():
            masks['dil'] = mask_dil

    if size:
        mask_filter, mask_mito = sizefilter_mask(mask, size)
        mask = mask_filter
        if 'mito' in outpaths.keys():
            masks['mito'] = mask_mito

    masks['out'] = mask

    return masks


def get_outpaths(h5path_out, save_steps, dilation, size):

    outpaths = {'out': h5path_out}
    if save_steps:
        outpaths['raw'] = ''
        if dilation > 0:
            outpaths['dil'] = ''
        if size > 0:
            outpaths['mito'] = ''
    outpaths = utils.gen_steps(outpaths, save_steps)

    return outpaths


def dilate_mask(mask, dilation=0, se=disk):

    mask = binary_dilation(mask, selem=se(dilation))

    return mask


def sizefilter_mask(mask, size=0):

    mask_filter = remove_small_objects(mask, min_size=size)
    mask_mito = np.logical_xor(mask, mask_filter)

    return mask_filter, mask_mito


def frange(x, y, jump):
    """Range for floats."""

    while x < y:
        yield x
        x += jump


if __name__ == "__main__":
    main(sys.argv[1:])
