#!/usr/bin/env python

"""Separate the myelin compartment into individual myelin sheaths.

"""

import sys
import argparse
import os

import numpy as np
from scipy.ndimage.morphology import grey_dilation, binary_dilation, binary_erosion
from scipy.special import expit
from scipy.ndimage import distance_transform_edt
from skimage import img_as_float
from skimage.measure import regionprops
from skimage.morphology import watershed

from wmem import parse, utils


def main(argv):
    """Separate the myelin compartment into individual myelin sheaths."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_separate_sheaths(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    separate_sheaths(
        args.inputfile,
        args.labelMM,
        args.maskWS,
        args.maskDS,
        args.maskMM,
        args.dilation_iterations,
        args.distance,
        args.sigmoidweighting,
        args.margin,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def separate_sheaths(
        h5path_in,
        h5path_lmm='',
        h5path_wsmask='',
        h5path_mask='',
        h5path_mmm='',
        MAdilation=0,
        h5path_dist='',
        sigmoidweighting=0,
        margin=50,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Separate the myelin compartment into individual myelin sheaths."""

    # check output paths
    outpaths = {'out': h5path_out,
                'wsmask': '',
                'madil{:02d}'.format(MAdilation): '',
                'distance_simple': '',
                'sheaths_simple': '',
                'distance_sigmod': '',
                }
    root, ds_main = outpaths['out'].split('.h5')
    for dsname, outpath in outpaths.items():
        grpname = ds_main + "_steps"
        outpaths[dsname] = os.path.join(root + '.h5' + grpname, dsname)
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
    ds_mma = ds_in[:] != 0

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    # load/calculate a mask to constrain the watershed in
    mask_mmregion = get_wsmask(outpaths, ds_mma, h5path_wsmask,
                               h5path_mask, h5path_mmm, MAdilation,
                               elsize, axlab)

    elsize_abs = np.absolute(elsize)
    seeds = grey_dilation(ds_in, size=(3, 3, 3))

    # load/calculate the distance transform
    if h5path_dist:
        h5file_dist, ds_dist, _, _ = utils.h5_load(h5path_dist)
    else:
        if sigmoidweighting:
            if h5path_lmm:
                ds_lmm = utils.h5_load(h5path_lmm, load_data=True)[0]
            else:
                ds_dist = distance_transform_edt(~ds_mma, sampling=elsize_abs)
                ds_dist = img_as_float(ds_dist)
                utils.save_step(outpaths, 'distance_simple',
                                ds_dist, elsize, axlab)
                ds_lmm = watershed(ds_dist, seeds, mask=mask_mmregion)
                utils.save_step(outpaths, 'sheaths_simple',
                                ds_lmm, elsize, axlab)
            ds_dist, _ = distance_transform_sw(ds_in, ds_lmm, elsize_abs,
                                               sigmoidweighting, margin)
            utils.save_step(outpaths, 'distance_sigmod',
                            ds_dist, elsize, axlab)
            ds_out[:] = watershed(ds_dist, seeds, mask=mask_mmregion)
            # TODO: save median widths mw
        else:
            ds_dist = distance_transform_edt(~ds_mma, sampling=elsize_abs)
            utils.save_step(outpaths, 'distance_simple',
                            ds_dist, elsize, axlab)
            ds_out[:] = watershed(ds_dist, seeds, mask=mask_mmregion)

    # close and return
    h5file_in.close()
    if h5path_dist:
        h5file_dist.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def add_h5paths_out(outpaths, info):
    """Add dictionary keys for datasets to be written."""

    (h5path_wsmask, MAdilation,
     h5path_dist, h5path_lmm, sigmoidweighting) = info

    root, ds_main = outpaths['out'].split('.h5')
    for dsname, outpath in outpaths.items():
        grpname = ds_main + "_steps"
        outpaths[dsname] = os.path.join(root + '.h5' + grpname, dsname)

    if not h5path_wsmask:
        outpaths['wsmask'] = ''
        if MAdilation:
            outpaths['madil{:02d}'.format(MAdilation)] = ''
    if not h5path_dist:
        if sigmoidweighting:
            if not h5path_lmm:
                outpaths['distance_simple'] = ''
                outpaths['sheaths_simple'] = ''
            outpaths['distance_sigmod'] = ''
        else:
            outpaths['distance_simple'] = ''

    return outpaths


def get_wsmask(outpaths, ds_mma,
               h5path_wsmask='', h5path_mds='', h5path_mmm='',
               MAdilation=0, elsize=None, axlab=None):
    """Load or construct a mask of the myelin region."""

    if h5path_wsmask:
        mask_mmregion = utils.h5_load(h5path_wsmask, load_data=True,
                                      dtype='bool')[0]
    else:
        mask_mmregion = construct_wsmask(outpaths, ds_mma,
                                         h5path_mds, h5path_mmm, MAdilation,
                                         elsize, axlab)

    return mask_mmregion


def construct_wsmask(outpaths, ds_mma, h5path_mds='', h5path_mmm='',
                     MAdilation=0, elsize=None, axlab=None):
    """Construct a mask of valid myelin voxels."""

    mask_mmregion = np.ones_like(ds_mma, dtype='bool')
    if MAdilation:
        mask_distance = binary_dilation(ds_mma, iterations=MAdilation)
        np.logical_and(mask_mmregion, mask_distance, mask_mmregion)
        mask_mmregion = mask_mmregion - ds_mma
        utils.save_step(outpaths, 'madil{:02d}'.format(MAdilation),
                  mask_distance.astype('bool'), elsize, axlab)
    if h5path_mmm:
        ds_mmm = utils.h5_load(h5path_mmm, load_data=True, dtype='bool')[0]
        np.logical_and(mask_mmregion, ds_mmm, mask_mmregion)
    if h5path_mds:
        ds_mds = utils.h5_load(h5path_mds, load_data=True, dtype='bool')[0]
        np.logical_and(mask_mmregion, ds_mds, mask_mmregion)

    utils.save_step(outpaths, 'wsmask',
                    mask_mmregion.astype('bool'), elsize, axlab)

    return mask_mmregion


def distance_transform_sw(labelMA, labelMM, elsize, weight=1, margin=50):
    """Calculate the sum of sigmoid-weighted distance transforms."""

    distsum = np.ones_like(labelMM, dtype='float')
    medwidth = {}

    dims = labelMM.shape
    labelMF = labelMA + labelMM

    rp = regionprops(labelMA)

    for prop in rp:
#         print(prop.label)
        if len(prop.bbox) > 4:
            z, y, x, Z, Y, X = tuple(prop.bbox)
            z = max(0, z - margin)
            Z = min(dims[0], Z + margin)
        else:
            y, x, Y, X = tuple(prop.bbox)
            z = 0
            Z = 1

        y = max(0, y - margin)
        x = max(0, x - margin)
        Y = min(dims[1], Y + margin)
        X = min(dims[2], X + margin)

        MAregion = labelMA[z:Z, y:Y, x:X]
        MFregion = labelMF[z:Z, y:Y, x:X]
        maskMA = MAregion == prop.label
        maskMF = MFregion == prop.label

        dist = distance_transform_edt(~maskMA, sampling=elsize)

        rim = maskMF - binary_erosion(maskMF, border_value=1)

        medwidth[prop.label] = np.median(dist[rim])
        weighteddist = expit(dist / (weight * medwidth[prop.label]))
        distsum[z:Z, y:Y, x:X] = np.minimum(distsum[z:Z, y:Y, x:X],
                                            weighteddist)

    return distsum, medwidth


if __name__ == "__main__":
    main(sys.argv)
