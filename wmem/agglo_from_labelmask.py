#!/usr/bin/env python

"""Apply mapping of labelsets to a labelvolume.

"""

import sys
import argparse

import numpy as np
from skimage.measure import regionprops

from wmem import parse, utils


def main(argv):
    """Apply mapping of labelsets to a labelvolume."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_agglo_from_labelmask(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    agglo_from_labelmask(
        args.inputfile,
        args.labelvolume,
        args.ratio_threshold,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def agglo_from_labelmask(
        h5path_in,
        h5path_lv='',
        ratio_threshold=0,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Apply mapping of labelsets to a labelvolume."""

    # check output paths
    outpaths = {'out': h5path_out}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
    h5file_lv, ds_lv, _, _ = utils.h5_load(h5path_lv)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    ulabels = np.unique(ds_in)
    maxlabel = np.amax(ulabels)
    print("number of labels in watershed: {:d}".format(maxlabel))

    fwmap = np.zeros(maxlabel + 1, dtype='i')

    areas_ws = np.bincount(ds_in.ravel())

    labelsets = {}
    rp_lw = regionprops(ds_lv, ds_in)
    for prop in rp_lw:

        maskedregion = prop.intensity_image[prop.image]
        counts = np.bincount(maskedregion)
        svoxs_in_label = [l for sl in np.argwhere(counts) for l in sl]

        ratios_svox_in_label = [float(counts[svox]) / float(areas_ws[svox])
                                for svox in svoxs_in_label]
        fwmask = np.greater(ratios_svox_in_label, ratio_threshold)
        labelset = np.array(svoxs_in_label)[fwmask]
        labelsets[prop.label] = set(labelset) - set([0])

    basepath = h5path_in.split('.h5/')[0]
    utils.write_labelsets(labelsets, basepath + "_svoxsets",
                          filetypes=['pickle'])

    ds_out[:] = utils.forward_map(np.array(fwmap), ds_in, labelsets)

    # close and return
    h5file_in.close()
    h5file_lv.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


if __name__ == "__main__":
    main(sys.argv[1:])
