#!/usr/bin/env python

"""Apply mapping of labelsets to a labelvolume.

"""

import sys
import argparse

import numpy as np
from skimage.measure import regionprops

from wmem import parse, utils, LabelImage


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
        args.oversegmentation,
        args.ratio_threshold,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def agglo_from_labelmask(
        image_in,
        oversegmentation,
        ratio_threshold=0,
        outputpath='',
        save_steps=False,
        protective=False,
        ):
    """Apply mapping of labelsets to a labelvolume."""

    axons = utils.get_image(image_in, imtype='Label')
    svoxs = utils.get_image(oversegmentation, imtype='Label')

    mo = LabelImage(outputpath, protective=protective,
                    **svoxs.get_image_props())
    mo.create()

    areas_svoxs = np.bincount(svoxs.ds[:].ravel().astype('int64'))
    rp = regionprops(axons.ds, svoxs.ds)
    labelsets = {}
    for axon in rp:
#         FIXME: double assignments?
        labelsets[axon.label] = assign_supervoxels_to_axon(axon,
                                                           areas_svoxs,
                                                           ratio_threshold)

    mo.write_labelsets(labelsets)
    mo.ds[:] = svoxs.forward_map(labelsets=labelsets, from_empty=True)

    svoxs.close()
    axons.close()
    mo.close()

    return mo


def assign_supervoxels_to_axon(axon, areas_svox, ratio_threshold):
    """"""

    svoxs_in_label = axon.intensity_image[axon.image]
    areas_svoxs_in_label = np.bincount(svoxs_in_label.astype('int64'))
    list_svoxs_in_label = [l for sl in np.argwhere(areas_svoxs_in_label)
                           for l in sl]

    ratios_svox_in_label = [float(areas_svoxs_in_label[svox]) / float(areas_svox[svox])
                            for svox in list_svoxs_in_label]

    # return set of supervoxels larger than threshold
    fwmask = np.greater(ratios_svox_in_label, ratio_threshold)
    labelset = set(np.array(list_svoxs_in_label)[fwmask]) - set([0])

    return labelset


if __name__ == "__main__":
    main(sys.argv[1:])
