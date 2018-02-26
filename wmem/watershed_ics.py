#!/usr/bin/env python

"""Perform watershed on the intracellular space compartments.

"""

import sys
import argparse

import numpy as np
from scipy.ndimage import label
from skimage.morphology import watershed, remove_small_objects

from wmem import parse, utils


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
        args.masks,
        args.seedimage,
        args.seed_size,
        args.lower_threshold,
        args.upper_threshold,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def watershed_ics(
        h5path_in,
        masks=[],
        h5path_seeds='',
        seed_size=64,
        lower_threshold=None,
        upper_threshold=None,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Perform watershed on the intracellular space compartments."""

    # check output paths
    outpaths = {'out': h5path_out, 'seeds': h5path_seeds, 'mask': ''}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape[:3], 'uint32',
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    # load/generate the seeds
    if h5path_seeds:
        h5file_sds, ds_sds, _, _ = utils.h5_load(h5path_seeds)
    else:
        h5file_sds, ds_sds = utils.h5_write(None, ds_in.shape[:3], 'uint32',
                                            outpaths['seeds'],
                                            element_size_um=elsize,
                                            axislabels=axlab)

        lower_threshold = lower_threshold or np.amin(ds_in[:]) - 1
        upper_threshold = upper_threshold or np.amax(ds_in[:]) + 1
        ds_sds[:] = np.logical_and(ds_in[:] > lower_threshold,
                                   ds_in[:] <= upper_threshold)
        ds_sds[:], _ = label(ds_sds[:])
        ds_sds[:] = remove_small_objects(ds_sds[:], min_size=seed_size)
        """ NOTE:
        numpy/scipy/skimage inplace is not written when using hdf5
        Therefore, we cannot use:
        np.logical_and(ds_in[:] > lower_threshold,
                       ds_in[:] <= upper_threshold, ds_sds[:])
        num = label(ds_sds[:], output=ds_sds[:])
        remove_small_objects(ds_sds[:], min_size=seed_size, in_place=True)
        """

    # determine the mask
    mask = np.ones(ds_in.shape[:3], dtype='bool')
    mask = utils.string_masks(masks, mask)
    h5file_mask, ds_mask = utils.h5_write(None, ds_in.shape[:3], 'uint8',
                                          outpaths['mask'],
                                          element_size_um=elsize,
                                          axislabels=axlab)
    ds_mask[:] = mask
#     ds_mask[:].fill(1)
#     ds_mask[:] = utils.string_masks(masks, ds_mask[:])

    # perform the watershed
    ds_out[:] = watershed(-ds_in[:], ds_sds[:], mask=ds_mask[:])

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
        h5file_sds.close()
        h5file_mask.close()
    except (ValueError, AttributeError):
        return ds_out, ds_sds, ds_mask


if __name__ == "__main__":
    main(sys.argv[1:])
