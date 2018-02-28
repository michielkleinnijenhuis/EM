#!/usr/bin/env python

"""Create thresholded hard segmentations.

"""

import sys
import argparse
import os

import numpy as np

from skimage.morphology import remove_small_objects
from skimage.morphology import binary_dilation, ball, disk

from wmem import parse, utils


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
                args.inputmask,
                args.go2D,
                args.outputfile.format(thr),
                args.save_steps,
                args.protective,
                )
    else:
        prob2mask(
            args.inputfile,
            args.dataslices,
            args.lower_threshold,
            args.upper_threshold,
            args.size,
            args.dilation,
            args.inputmask,
            args.go2D,
            args.outputfile,
            args.save_steps,
            args.protective,
            )


def prob2mask(
        h5path_in,
        dataslices=None,
        lower_threshold=0,
        upper_threshold=1,
        size=0,
        dilation=0,
        h5path_mask=None,
        go2D=False,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Create thresholded hard segmentation."""

    # Check if any output paths already exist.
    outpaths = {'out': h5path_out, 'raw': '', 'mito': '', 'dil': ''}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # Load the (sliced) data.
    h5file_in, ds_in, es, al = utils.load(h5path_in)
    data, _, _, slices_out = utils.load_dataset(ds_in, dataslices=dataslices)
    inmask = utils.load(h5path_mask, load_data=True, dtype='bool',
                        dataslices=dataslices)[0]

    if data.ndim == 4:  # FIXME: generalize
#         data = np.squeeze(data)
        data = data[:, :, :, 0]
        outshape = ds_in.shape[:3]
        es = es[:3]
        al = al[:3]
        slices_out = slices_out[:3]
    else:
        outshape = ds_in.shape

    # Open the outputfile(s) for writing
    # and create the dataset(s) or output array(s).
    h5file_out, ds_out = utils.h5_write(None, outshape, 'uint8',
                                        outpaths['out'],
                                        element_size_um=es,
                                        axislabels=al)
    if save_steps:
        stepnames = ['raw', 'mito', 'dil']
        steps = [utils.h5_write(None, outshape, 'uint8',
                                outpaths[out],
                                element_size_um=es,
                                axislabels=al)
                 for out in stepnames]
        dss = [np.zeros_like(data)] * 3
    else:
        stepnames, steps, dss = [], [], []

    # Threshold (and dilate and filter) the data.
    if go2D:
        # process slicewise
        mask = np.zeros_like(data)
        for i, slc in enumerate(data):
            smf, sdss = process_slice(slc,
                                      lower_threshold, upper_threshold,
                                      size, dilation, disk, save_steps)
            mask[i, :, :] = smf
            for ds, sds in zip(dss, sdss):
                ds[i, :, :] = sds
    else:
        # process full input
        mask, dss = process_slice(data,
                                  lower_threshold, upper_threshold,
                                  size, dilation, ball, save_steps)

    # Apply additional mask.
    if inmask is not None:
        mask[~inmask] = False

    # Write the mask(s) to file.
    if '.h5' in h5path_out:
        utils.write_to_h5ds(ds_out, mask.astype('uint8'), slices_out)
        for step, ds in zip(steps, dss):
            utils.write_to_h5ds(step[1], ds.astype('uint8'), slices_out)
    else:  # write as tif/png/jpg series
        root, ext = os.path.splitext(outpaths['out'])
        utils.write_to_img(root, mask.astype('uint8'), al, 5, ext, 0)
        for stepname, ds in zip(stepnames, dss):
            root, ext = os.path.splitext(outpaths[stepname])
            utils.write_to_img(root, ds.astype('uint8'), al, 5, ext, 0)

    # Close the h5 file(s) or return the output array(s).
    try:
        h5file_in.close()
        h5file_out.close()
        for step in steps:
            step[0].close()
    except (ValueError, AttributeError):
        return mask, dss


def process_slice(data, lt, ut, size=0, dilation=0,
                  se=disk, save_steps=False):
    """Threshold and filter data."""

    mask_raw, mask_mito, mask_dil = None, None, None

    mask = np.logical_and(data > lt, data <= ut)
    mask_raw = mask.copy()

    if dilation:
        mask = binary_dilation(mask, selem=se(dilation))
        mask_dil = mask.copy()

    if size:
        mask_filter = remove_small_objects(mask, min_size=size)
        mask_mito = np.logical_xor(mask, mask_filter)
        mask = mask_filter

    if save_steps:
        return mask, (mask_raw, mask_mito, mask_dil)
    else:
        return mask, []


def frange(x, y, jump):
    """Range for floats."""

    while x < y:
        yield x
        x += jump


if __name__ == "__main__":
    main(sys.argv[1:])
