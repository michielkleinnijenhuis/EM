#!/usr/bin/env python

"""Apply mapping of labelsets to a labelvolume.

"""

import sys
import argparse

import numpy as np
from skimage.measure import regionprops

from wmem import parse, utils, wmeMPI, LabelImage


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
        args.usempi & ('mpi4py' in sys.modules),
        )


def agglo_from_labelmask(
        image_in,
        oversegmentation,
        ratio_threshold=0,
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Apply mapping of labelsets to a labelvolume."""

    mpi = wmeMPI(usempi)

    axons = utils.get_image(image_in, comm=mpi.comm, imtype='Label')
    svoxs = utils.get_image(oversegmentation, comm=mpi.comm, imtype='Label')

    props = svoxs.get_props(protective=protective)
    mo = LabelImage(outputpath, **props)
    mo.create(comm=mpi.comm)
#     mo2 = LabelImage(outputpath + '_svoxs', **props)
#     mo2.create(comm=mpi.comm)
#     mo3 = LabelImage(outputpath + '_axons', **props)
#     mo3.create(comm=mpi.comm)

    areas_svoxs = np.bincount(svoxs.ds[:].ravel().astype('int64'))
    rp = regionprops(axons.ds, svoxs.ds)
    labelsets = {}

    mpi.nblocks = len(rp)
    mpi.scatter_series(randomize=True)

    for i in mpi.series:
        axon = rp[i]
#         FIXME: double assignments?
        labelset = assign_supervoxels_to_axon(axon, areas_svoxs, ratio_threshold)

        # THIS bit was for testing probMA without threshold
#         svoxs_in_label = axon.intensity_image[axon.image]
#         areas_svoxs_in_label = np.bincount(svoxs_in_label.astype('int64'))
#         list_svoxs_in_label = [l for sl in np.argwhere(areas_svoxs_in_label)
#                                for l in sl]
#         labelset = set(list_svoxs_in_label) - set([0])

#         if labelset:
        N_svoxs = 0
        if len(labelset) > N_svoxs:
            labelsets[axon.label] = labelset

    comps = axons.split_path(outputpath)
    utils.dump_labelsets(labelsets, comps, mpi.rank)

    if mpi.enabled:
        mpi.comm.Barrier()

    if mpi.rank == 0:

        labelsets = utils.combine_labelsets({}, comps)
        mo.write(svoxs.forward_map(labelsets=labelsets, from_empty=True))

#         ls_axons = {lskey: set([lskey]) for lskey in labelsets.keys()}
#         svoxs_fw = svoxs.forward_map(labelsets=labelsets, from_empty=True)
#         mo2.write(svoxs_fw)
#         axons_fw = axons.forward_map(labelsets=ls_axons, from_empty=True)
#         mo3.write(axons_fw)
#         mo.write(np.maximum(svoxs_fw, axons_fw))

    svoxs.close()
    axons.close()
    mo.close()
#     mo2.close()
#     mo3.close()
# 
#     return mo


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
