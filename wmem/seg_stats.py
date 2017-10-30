#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

from skimage.measure import regionprops


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('--labelMA', nargs=2, default=[],
                        help='...')
    parser.add_argument('--labelMF', nargs=2, default=[],
                        help='...')
    parser.add_argument('--labelUA', nargs=2, default=[],
                        help='...')
    parser.add_argument('--stats', nargs='*', default=[],
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelMA = args.labelMA
    labelMF = args.labelMF
    labelUA = args.labelUA
    stats = args.stats


    if labelUA:
        name = 'UA'
        UAprops = get_fibre_stats(datadir, dset_name, labelUA, name, stats)
        save_stats(datadir, name, UAprops)

    if labelMA:
        name = 'MA'
        MAprops = get_fibre_stats(datadir, dset_name, labelMA, name, stats)
        save_stats(datadir, name, MAprops)
        MA_area = MAprops['area']

    if labelMF:
        name = 'MF'
        MFprops = get_fibre_stats(datadir, dset_name, labelMF, name, stats)
        MF_area = MFprops['area']

    if labelMA and labelMF:
        # per-axon g-ratios
        gratios = np.empty(MA_area.shape[1])
        gratios[:] = np.NAN

        MA_area_tot = np.nansum(MA_area, axis=0)
        MF_area_tot = np.nansum(MF_area, axis=0)
        mask = (MA_area_tot > 0) & (MF_area_tot > 0)

        MM_area_tot = MF_area_tot[mask] - MA_area_tot[mask]
        gratios = np.sqrt(1 - MM_area_tot / MF_area_tot[mask])
        np.savetxt(os.path.join(datadir, 'stats_gratios.txt'), gratios[mask])

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
        fname = os.path.join(datadir, 'stats_gratios_slcws.txt')
        np.savetxt(fname, np.array(gratios))

    # TODO: use the mask as in filter_NoR


# ========================================================================== #
# function defs
# ========================================================================== #


def get_fibre_stats(datadir, dset_name, labelvolume, name, stats=[]):

    LV = loadh5(datadir, dset_name + labelvolume[0],
                fieldname=labelvolume[1])[0]

    ulabels = np.unique(LV)
    Nlabels = len(ulabels) -1
    print("number of %s labels: %d" % (name, Nlabels))

    props = {}
    props['area'] = np.empty([LV.shape[0], Nlabels], dtype='float')
    props['area'][:, :] = np.NAN
    if 'AD' in stats:
        props['AD'] = np.copy(props['area'])
    if 'centroid' in stats:
        props['centroid'] = np.empty([LV.shape[0], Nlabels, 2], dtype='float')
        props['centroid'][:, :, :] = np.NAN
    if 'eccentricity' in stats:
        props['eccentricity'] = np.copy(props['area'])
    if 'solidity' in stats:
        props['solidity'] = np.copy(props['area'])

    for i in range(0, LV.shape[0]):
        print(i)
        rp = regionprops(LV[i,:,:])
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

    save_stats(datadir, name, props)

    return props


def save_stats(datadir, name, props):

    fname = os.path.join(datadir, 'stats_%s_slcws_area.txt' % name)
    np.savetxt(fname, np.array(props['area']))
    if 'AD' in props.keys():
        fname = os.path.join(datadir, 'stats_%s_slcws_AD.txt' % name)
        np.savetxt(fname, np.array(props['AD']))
    if 'centroid' in props.keys():
        fname = os.path.join(datadir, 'stats_%s_slcws_centroid_y.txt' % name)
        np.savetxt(fname, np.array(props['centroid'][:,:,0]))
        fname = os.path.join(datadir, 'stats_%s_slcws_centroid_x.txt' % name)
        np.savetxt(fname, np.array(props['centroid'][:,:,1]))
    if 'eccentricity' in props.keys():
        fname = os.path.join(datadir, 'stats_%s_slcws_eccentricity.txt' % name)
        np.savetxt(fname, np.array(props['eccentricity']))
    if 'solidity' in props.keys():
        fname = os.path.join(datadir, 'stats_%s_slcws_solidity.txt' % name)
        np.savetxt(fname, np.array(props['solidity']))


def loadh5(datadir, dname, fieldname='stack', dtype=None):
    """Load a h5 stack."""

    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')

    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:, :]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:, :, :]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:, :, :, :]

    element_size_um, axislabels = get_h5_attributes(f[fieldname])

    f.close()

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

    return stack, element_size_um, axislabels


def writeh5(stack, datadir, fp_out, fieldname='stack',
            dtype='uint16', element_size_um=None, axislabels=None):
    """Write a h5 stack."""

    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")

    if len(stack.shape) == 2:
        g[fieldname][:, :] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:, :, :] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:, :, :, :] = stack

    write_h5_attributes(g[fieldname], element_size_um, axislabels)

    g.close()


def get_h5_attributes(stack):
    """Get attributes from a stack."""

    element_size_um = axislabels = None

    if 'element_size_um' in stack.attrs.keys():
        element_size_um = stack.attrs['element_size_um']

    if 'DIMENSION_LABELS' in stack.attrs.keys():
        axislabels = stack.attrs['DIMENSION_LABELS']

    return element_size_um, axislabels


def write_h5_attributes(stack, element_size_um=None, axislabels=None):
    """Write attributes to a stack."""

    if element_size_um is not None:
        stack.attrs['element_size_um'] = element_size_um

    if axislabels is not None:
        for i, l in enumerate(axislabels):
            stack.dims[i].label = l


if __name__ == "__main__":
    main(sys.argv)
