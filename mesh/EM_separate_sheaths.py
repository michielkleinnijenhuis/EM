#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.morphology import watershed
from scipy.ndimage.morphology import grey_dilation, binary_erosion
from scipy.special import expit
from scipy.ndimage import distance_transform_edt


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume', nargs=2,
                        default=['_labelMA', 'stack'],
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_labelMM_ws', 'stack'],
                        help='...')
    parser.add_argument('--maskDS', nargs=2, default=['_maskDS', 'stack'],
                        help='...')
    parser.add_argument('--maskMM', nargs=2, default=['_maskMM', 'stack'],
                        help='...')
    parser.add_argument('--maskMA', nargs=2, default=None,
                        help='...')
    parser.add_argument('--dist', nargs=2, default=None,
                        help='...')
    parser.add_argument('-w', '--sigmoidweighting', action='store_true',
                        help='...')
    parser.add_argument('-d', '--distancefilter', action='store_true',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    outpf = args.outpf
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMA = args.maskMA
    dist = args.dist
    sigmoidweighting = args.sigmoidweighting
    distancefilter = args.distancefilter

    MA, elsize, al = loadh5(datadir, dset_name + labelvolume[0],
                            fieldname=labelvolume[1])
    maskDS = loadh5(datadir, dset_name + maskDS[0],
                    fieldname=maskDS[1], dtype='bool')[0]
    maskMM = loadh5(datadir, dset_name + maskMM[0],
                    fieldname=maskMM[1], dtype='bool')[0]
    if maskMA is not None:
        maskMA = loadh5(datadir, dset_name + maskMA[0],
                        fieldname=maskMA[1], dtype='bool')[0]
    else:
        maskMA = MA != 0

    # watershed on simple distance transform
    if dist is not None:
        distance = loadh5(datadir, dset_name + dist[0],
                          fieldname=dist[1])[0]
    else:
        abs_elsize = np.absolute(elsize)
        distance = distance_transform_edt(~maskMA, sampling=abs_elsize)
        sname = dset_name + outpf[0] + '_dist'
        writeh5(distance, datadir, sname, fieldname=outpf[1],
                dtype='float', element_size_um=elsize, axislabels=al)

    seeds = grey_dilation(MA, size=(3,3,3))
    mask = np.logical_and(maskMM, maskDS)
    MM = watershed(distance, seeds, mask=mask)
    sname = dset_name + outpf[0] + '_ws'
    writeh5(MM, datadir, sname, fieldname=outpf[1],
            dtype='int32', element_size_um=elsize, axislabels=al)

    # watershed on sigmoid-modulated distance transform
    if sigmoidweighting:

        distsum, lmask = sigmoid_weighted_distance(MM, MA, abs_elsize)
        sname = dset_name + outpf[0] + '_distsum'
        writeh5(distsum, datadir, sname,
                dtype='float', element_size_um=elsize, axislabels=al)

        MM = watershed(distsum, grey_dilation(MA, size=(3,3,3)), 
                       mask=np.logical_and(maskMM, maskDS))
        sname = dset_name + outpf[0] + '_sw'
        writeh5(MM, datadir, sname,
                dtype='int32', element_size_um=elsize, axislabels=al)

    elif distancefilter:  # TODO simple distance th

        lmask = np.zeros((MM.shape[0], MM.shape[1], MM.shape[2], 
                          len(np.unique(MA)[1:])), dtype='bool')

    if distancefilter:  # very mem-intensive

        for i,l in enumerate(np.unique(MA)[1:]):
            MM[np.logical_and(lmask[:,:,:,i], MM==l)] = 0
        sname = dset_name + outpf[0] + '_df'
        writeh5(MM, datadir, sname,
                dtype='int32', element_size_um=elsize, axislabels=al)


# ========================================================================== #
# function defs
# ========================================================================== #


def sigmoid_weighted_distance(MM, MA, elsize):
    """"""

    lmask = np.zeros((MM.shape[0], MM.shape[1], MM.shape[2], 
                      len(np.unique(MA)[1:])), dtype='bool')
    distsum = np.ones_like(MM, dtype='float')
    medwidth = {}
    for i,l in enumerate(np.unique(MA)[1:]):  # TODO: implement mpi?
        print(i,l)
        dist = distance_transform_edt(MA!=l, sampling=elsize)
        # get the median distance at the outer rim:
        MMfilled = MA + MM
        binim = MMfilled == l
        rim = np.logical_xor(binary_erosion(binim), binim)
        medwidth[l] = np.median(dist[rim])
        # labelmask for voxels further than nmed medians from the object (mem? write to disk?)
        nmed = 2  # TODO: make into argument  # to measured in um for low res processing
        maxdist = nmed * medwidth[l]
        lmask[:,:,:,i] = dist > maxdist
        # median width weighted sigmoid transform on distance function
        weighteddist = expit(dist/medwidth[l])  # TODO: create more pronounced transform
        distsum = np.minimum(distsum, weighteddist)

    return distsum, lmask


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
