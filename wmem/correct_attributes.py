#!/usr/bin/env python

"""Correct attributes of a h5 dataset.

"""

import sys
import argparse
import numpy as np

from wmem import parse, utils


def main(argv):
    """Correct attributes of a h5 dataset."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_correct_attributes(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    correct_attributes(
        args.inputfile,
        args.auxfile,
        args.outlayout,
        args.element_size_um,
        )


def correct_attributes(
        h5path_in,
        h5path_aux='',
        axlab='',
        elsize=[],
        ):
    """Correct attributes of a h5 dataset."""

    h5file_in, ds_in, _, _ = utils.h5_load(h5path_in)

    if h5path_aux:
        h5file_aux, _, h5elsize, h5axlab = utils.h5_load(h5path_aux)
        elsize = elsize or h5elsize
        axlab = axlab or h5axlab

    # FIXME: sloppy insertion
    if len(elsize) < ds_in.ndim:
        elsize = np.append(elsize, 1)
    if len(axlab) < ds_in.ndim:
        axlab.append('c')

    utils.h5_write_attributes(ds_in, element_size_um=elsize, axislabels=axlab)

    try:
        h5file_in.close()
        h5file_aux.close()
    except (ValueError, AttributeError):
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
