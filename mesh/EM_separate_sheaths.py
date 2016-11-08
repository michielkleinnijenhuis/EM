#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import watershed
from scipy.ndimage.morphology import grey_dilation, binary_dilation, binary_erosion
from scipy.special import expit
from scipy.ndimage import distance_transform_edt


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelMA', nargs=2,
                        default=['_labelMA', 'stack'],
                        help='...')
    parser.add_argument('-L', '--labelMM', nargs=2, default=[],
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_labelMM', 'stack'],
                        help='...')
    parser.add_argument('--maskDS', nargs=2, default=['_maskDS', 'stack'],
                        help='...')
    parser.add_argument('--maskMM', nargs=2, default=['_maskMM', 'stack'],
                        help='...')
    parser.add_argument('--maskMA', nargs=2, default=[],
                        help='...')
    parser.add_argument('--MAdilation', type=int, default=0,
                        help='...')
    parser.add_argument('--dist', nargs=2, default=[],
                        help='...')
    parser.add_argument('-w', '--sigmoidweighting', action='store_true',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelMA = args.labelMA
    labelMM = args.labelMM
    outpf = args.outpf
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMA = args.maskMA
    MAdilation = args.MAdilation
    dist = args.dist
    sigmoidweighting = args.sigmoidweighting

    MA, elsize, al = loadh5(datadir, dset_name + labelMA[0],
                            fieldname=labelMA[1])
    if maskMA:
        maskMA = loadh5(datadir, dset_name + maskMA[0],
                        fieldname=maskMA[1], dtype='bool')[0]
    else:
        maskMA = MA != 0

    if MAdilation:
        # mask to perform constrain the watershed in
        maskDS = loadh5(datadir, dset_name + maskDS[0],
                        fieldname=maskDS[1], dtype='bool')[0]
        maskMM = loadh5(datadir, dset_name + maskMM[0],
                        fieldname=maskMM[1], dtype='bool')[0]
        mask = np.logical_and(maskMM, maskDS)
        maskdist = binary_dilation(maskMA, iterations=MAdilation)
        aname = dset_name + outpf[0] + '_MAdilation'
        writeh5(maskdist, datadir, aname, fieldname=outpf[1],
                dtype='uint8', element_size_um=elsize, axislabels=al)
        np.logical_and(mask, maskdist, mask)
        aname = dset_name + outpf[0] + '_MM_wsmask'
        writeh5(mask, datadir, aname, fieldname=outpf[1],
                dtype='uint8', element_size_um=elsize, axislabels=al)

    # watershed on simple distance transform
    if dist:
        distance = loadh5(datadir, dset_name + dist[0],
                          fieldname=dist[1])[0]
    else:
        abs_elsize = np.absolute(elsize)
        distance = distance_transform_edt(~maskMA, sampling=abs_elsize)
        sname = dset_name + outpf[0] + '_dist'
        writeh5(distance, datadir, sname, fieldname=outpf[1],
                dtype='float', element_size_um=elsize, axislabels=al)

    # perform the watershed
    if labelMM:
        MM = loadh5(datadir, dset_name + labelMM[0],
                    fieldname=labelMM[1])
    else:
        # prepare the seeds to overlap with mask
        seeds = grey_dilation(MA, size=(3,3,3))
        MM = watershed(distance, seeds, mask=mask)
        sname = dset_name + outpf[0] + '_ws'
        writeh5(MM, datadir, sname, fieldname=outpf[1],
                dtype='int32', element_size_um=elsize, axislabels=al)

    # watershed on sigmoid-modulated distance transform
    if sigmoidweighting:

        distsum = sigmoid_weighted_distance(MM, MA, abs_elsize)

        sname = dset_name + outpf[0] + '_distsum'
        writeh5(distsum, datadir, sname,
                dtype='float', element_size_um=elsize, axislabels=al)

        MM = watershed(distsum, grey_dilation(MA, size=(3,3,3)), 
                       mask=np.logical_and(maskMM, maskDS))

        sname = dset_name + outpf[0] + '_sw'
        writeh5(MM, datadir, sname,
                dtype='int32', element_size_um=elsize, axislabels=al)


# ========================================================================== #
# function defs
# ========================================================================== #


def sigmoid_weighted_distance(MM, MA, elsize):
    """"""

    distsum = np.ones_like(MM, dtype='float')
    medwidth = {}

    dims = MM.shape
    # TODO: check if MM is filled already!
#     MMfilled = MA + MM
    MMfilled = MM

    rp = regionprops(MA, MMfilled)

    for prop in rp:
        l = prop.label
        print(l)
        z, y, x, Z, Y, X = tuple(prop.bbox)
        m = 10
        z = max(0, z - m)
        y = max(0, y - m)
        x = max(0, x - m)
        Z = min(dims[0], Z + m)
        Y = min(dims[1], Y + m)
        X = min(dims[2], X + m)

        mask = MA[z:Z, y:Y, x:X] == l

        dist = distance_transform_edt(~mask, sampling=elsize)

        binim = MMfilled[z:Z, y:Y, x:X][mask] == l
        rim = np.logical_xor(binary_erosion(binim), binim)
        medwidth[l] = np.median(dist[rim])

        # median width weighted sigmoid transform on distance function
        weighteddist = expit(dist/medwidth[l])  
        # TODO: create more pronounced transform?
        distsum[z:Z, y:Y, x:X] = np.minimum(distsum[z:Z, y:Y, x:X],
                                            weighteddist)

    return distsum


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
