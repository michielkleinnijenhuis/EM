#!/usr/bin/env python

"""Create thresholded hard segmentations.

"""

import sys
import argparse

import numpy as np

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
                args.blockreduce,
                args.outputfile.format(thr),
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
            args.blockreduce,
            args.outputfile,
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
        h5path_out='',
        protective=False,
        ):
    """Create thresholded hard segmentation."""

    if bool(h5path_out) and ('.h5' in h5path_out):
        status, info = utils.h5_check(h5path_out, protective)
        print(info)
        if status == "CANCELLED":
            return

    # open data for reading
    h5file_in, ds_in, es, al = utils.load(h5path_in)
    data, _, _, slices_out = utils.load_dataset(
        ds_in, es, al, al, dtype='', dataslices=dataslices)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, 'uint8',
                                        h5path_out,
                                        element_size_um=es,
                                        axislabels=al)

    mask = np.logical_and(data > lower_threshold,
                          data <= upper_threshold)

    if size:
        from skimage.morphology import remove_small_objects
        remove_small_objects(mask, min_size=size, in_place=True)

    if dilation:
        from skimage.morphology import binary_dilation, ball
        mask = binary_dilation(mask, selem=ball(dilation))

    if h5path_mask is not None:
        inmask = utils.load(h5path_mask, load_data=True, dtype='bool',
                            dataslices=dataslices)[0]
        mask[~inmask] = False

    if '.h5' in h5path_out:
        utils.write_to_h5ds(ds_out, mask.astype('uint8'), slices_out)
    elif h5path_out.endswith('.tif'):
        utils.imf_write(h5path_out, mask.astype('uint8'))

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return mask


def frange(x, y, jump):
    """Range for floats."""
    while x < y:
        yield x
        x += jump


if __name__ == "__main__":
    main(sys.argv[1:])
