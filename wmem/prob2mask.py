#!/usr/bin/env python

"""Create thresholded hard segmentations.

"""

import sys
import argparse

import numpy as np
from skimage.morphology import remove_small_objects, binary_dilation, ball
from skimage.measure import block_reduce

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
            mask = prob2mask(
                args.inputfile,
                args.channel,
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
            args.channel,
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
        h5path_probs,
        channel=None,
        lower_threshold=0,
        upper_threshold=1,
        size=0,
        dilation=0,
        h5path_mask=None,
        blockreduce=None,
        h5path_out='',
        protective=False,
        ):
    """Create thresholded hard segmentation."""

    if bool(h5path_out) and ('.h5' in h5path_out):  # FIXME: do check for other output formats
        status, info = utils.h5_check(h5path_out, protective)
        print(info)
        if status == "CANCELLED":
            return

    prob, es, al = utils.load(h5path_probs,
                              channels=channel,
                              load_data=True)

    mask = np.logical_and(prob > lower_threshold,
                          prob <= upper_threshold)

    if size:
        remove_small_objects(mask, min_size=size, in_place=True)

    if dilation:
        mask = binary_dilation(mask, selem=ball(dilation))

    if h5path_mask is not None:
        inmask = utils.load(h5path_mask, dtype='bool',
                            load_data=True)[0]
        mask[~inmask] = False

    if blockreduce:
        mask = block_reduce(mask, block_size=tuple(blockreduce), func=np.amax)
        es = [e * b for e, b in zip(es, blockreduce)]

    if '.h5' in h5path_out:
        utils.h5_write(mask.astype('uint8'), mask.shape, 'uint8',
                       h5path_out,
                       element_size_um=es, axislabels=al)
    elif h5path_out.endswith('.tif'):
        utils.imf_write(h5path_out, mask.astype('uint8'))

    return mask


def frange(x, y, jump):
    """Range for floats."""
    while x < y:
        yield x
        x += jump


if __name__ == "__main__":
    main(sys.argv[1:])
