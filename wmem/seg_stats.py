#!/usr/bin/env python

"""Compute segmentation statistics.

"""

import os
import sys
import argparse

import h5py
import numpy as np

from skimage.measure import regionprops

from wmem import parse, utils


def main(argv):
    """Compute segmentation statistics."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_seg_stats(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    seg_stats(
        args.h5path_labelMA,
        args.h5path_labelMF,
        args.h5path_labelUA,
        args.stats,
        args.outputbasename,
        args.protective,
        )


def seg_stats(
        labelMA='',
        labelMF='',
        labelUA='',
        stats=['area',
               'AD',
               'centroid',
               'eccentricity',
               'solidity'],
        outputbasename='',
        protective=False,
        ):

    if labelUA:
        name = 'UA'
        UAprops = get_fibre_stats(labelUA, stats)
        if not outputbasename:
            outbase = labelUA.split('.h5')[0] + '_stats_UA_'
        save_stats(outbase, UAprops)

    if labelMA:
        name = 'MA'
        MAprops = get_fibre_stats(labelMA, stats)
        if not outputbasename:
            outbase = labelMA.split('.h5')[0] + '_stats_MA_'
        save_stats(outbase, MAprops)
        MA_area = MAprops['area']

    if labelMF:
        name = 'MF'
        MFprops = get_fibre_stats(labelMF, stats)
        if not outputbasename:
            outbase = labelMF.split('.h5')[0] + '_stats_MF_'
        save_stats(outbase, MFprops)
        MF_area = MFprops['area']

    if labelMA and labelMF:

        if not outputbasename:
            outbase = labelMF.split('.h5')[0] + '_stats_GR_'

        # per-axon g-ratios
        gratios = np.empty(MA_area.shape[1])
        gratios[:] = np.NAN

        MA_area_tot = np.nansum(MA_area, axis=0)
        MF_area_tot = np.nansum(MF_area, axis=0)
        mask = (MA_area_tot > 0) & (MF_area_tot > 0)

        MM_area_tot = MF_area_tot[mask] - MA_area_tot[mask]
        gratios = np.sqrt(1 - MM_area_tot / MF_area_tot[mask])
        fname = outbase + 'gratios.txt'
        np.savetxt(fname, gratios[mask])

        # slicewise g-ratios
        gratios = np.copy(MF_area)
        gratios[:, :] = np.NAN
        for i in range(0, MA_area.shape[0]):
            for j in range(0, MA_area.shape[1]):
                try:
                    MM_area = MF_area[i, j] - MA_area[i, j]
                    gratios[i, j] = np.sqrt(1 - float(MM_area) /
                                            float(MF_area[i, j]))
                except:
                    pass
        fname = outbase + 'gratios_slcws.txt'
        np.savetxt(fname, np.array(gratios))

    # TODO: use the mask as in filter_NoR

    return


def get_fibre_stats(h5path_in, stats=[]):
    """Calculate statistics."""

    ds_in = utils.h5_load(h5path_in, load_data=True)[0]

    ulabels = np.unique(ds_in)
    Nlabels = len(ulabels) - 1
#     print("number of %s labels: %d" % (name, Nlabels))

    props = {}
    props['area'] = np.empty([ds_in.shape[0], Nlabels], dtype='float')
    props['area'][:, :] = np.NAN
    if 'AD' in stats:
        props['AD'] = np.copy(props['area'])
    if 'centroid' in stats:
        props['centroid'] = np.empty([ds_in.shape[0], Nlabels, 2], dtype='float')
        props['centroid'][:, :, :] = np.NAN
    if 'eccentricity' in stats:
        props['eccentricity'] = np.copy(props['area'])
    if 'solidity' in stats:
        props['solidity'] = np.copy(props['area'])

    for i in range(0, ds_in.shape[0]):
        rp = regionprops(ds_in[i, :, :])
        areas = {prop.label: prop.area for prop in rp}
        if 'AD' in stats:
            eqdia = {prop.label: prop.equivalent_diameter for prop in rp}
        if 'centroid' in stats:
            centr = {prop.label: prop.centroid for prop in rp}
        if 'eccentricity' in stats:
            ecctr = {prop.label: prop.eccentricity for prop in rp}
        if 'solidities' in stats:
            solid = {prop.label: prop.solidity for prop in rp}

        for j, l in enumerate(ulabels[1:]):
            try:
                props['area'][i, j] = areas[l]
            except:
                pass
            if 'AD' in stats:
                try:
                    props['AD'][i, j] = eqdia[l]
                except:
                    pass
            if 'centroid' in stats:
                try:
                    props['centroid'][i, j, :] = centr[l]
                except:
                    pass
            if 'eccentricity' in stats:
                try:
                    props['eccentricity'][i, j] = ecctr[l]
                except:
                    pass
            if 'solidity' in stats:
                try:
                    props['solidity'][i, j] = solid[l]
                except:
                    pass

    return props


def save_stats(outputbasename, props):
    """Save statistics."""

    fname = outputbasename + 'area.txt'
    np.savetxt(fname, np.array(props['area']))
    if 'AD' in props.keys():
        fname = outputbasename + 'AD.txt'
        np.savetxt(fname, np.array(props['AD']))
    if 'centroid' in props.keys():
        fname = outputbasename + 'slcws_centroid_y.txt'
        np.savetxt(fname, np.array(props['centroid'][:, :, 0]))
        fname = outputbasename + 'slcws_centroid_x.txt'
        np.savetxt(fname, np.array(props['centroid'][:, :, 1]))
    if 'eccentricity' in props.keys():
        fname = outputbasename + 'slcws_eccentricity.txt'
        np.savetxt(fname, np.array(props['eccentricity']))
    if 'solidity' in props.keys():
        fname = outputbasename + 'slcws_solidity.txt'
        np.savetxt(fname, np.array(props['solidity']))


if __name__ == "__main__":
    main(sys.argv)
