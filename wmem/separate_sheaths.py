#!/usr/bin/env python

"""Separate the myelin compartment into individual myelin sheaths.

"""

import sys
import argparse
import os
import pickle

import numpy as np
from scipy.ndimage.morphology import (grey_dilation,
                                      binary_dilation,
                                      binary_erosion)
from scipy.special import expit
from scipy.ndimage import distance_transform_edt
from skimage import img_as_float
from skimage.measure import regionprops
from skimage.morphology import watershed

from wmem import parse, utils, wmeMPI, Image, MaskImage, LabelImage
from wmem.merge_slicelabels import generate_anisotropic_selem


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
        args.dataslices,
        args.labelMM,
        args.maskWS,
        args.maskDS,
        args.maskMM,
        args.dilation_iterations,
        args.distance,
        args.sigmoidweighting,
        args.margin,
        args.medwidth_file,
        args.outputfile,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def separate_sheaths(
        image_in,
        dataslices=None,
        h5path_lmm='',
        h5path_wsmask='',
        h5path_mds='',
        h5path_mmm='',
        MAdilation=[1, 7, 7],
        h5path_dist='',
        sigmoidweighting=0,
        margin=50,
        medwidth_file='',
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Separate the myelin compartment into individual myelin sheaths."""

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    ma = utils.get_image(image_in, comm=mpi.comm)
    mask_ma = ma.ds[:] != 0
    seeds = grey_dilation(ma.ds, size=(3, 3, 3))

    # Determine the properties of the output dataset.
    outpaths = get_outpaths(outputpath, save_steps, sigmoidweighting)
    props = ma.get_props(protective=protective, dtype='uint64', squeeze=True)

    # TODO: prepare and pass all output volumes (wsmask, distance, sheaths)

    # load/calculate a mask to constrain the watershed in
#     if h5path_wsmask:
#         mask_ws = utils.get_image(h5path_wsmask)
#     else:
#         props['dtype'] = 'bool'
#         mask_ws = MaskImage(outpaths['wsmask'], **props)
#         mask_ws.create(comm=mpi.comm)
#         mask_ws.write(np.ones_like(mask_ma, dtype='bool'))  # FIXME: remove writing step

    dist_thr = 0.25  # TODO: make argument

    if h5path_lmm:

        label_mm = utils.get_image(h5path_lmm)

    else:

        if h5path_dist:
            dist = utils.get_image(h5path_dist)
        else:
            abs_es = np.absolute(props['elsize'])
            dist_data = distance_transform_edt(~mask_ma, sampling=abs_es)
            dist = write_dist(outpaths['distance'], props, dist_data)

        mask_ws = construct_wsmask(outpaths['wsmask'], props, dist,
                                   h5path_mds, h5path_mmm,
                                   MAdilation, dist_thr=dist_thr)

        label_mm = ws(outpaths['sheaths'], props, mask_ma, seeds, mask_ws, dist)

        dist.close()

    if sigmoidweighting:
        dsname = 'wsmask_sigmoid_{}'.format(sigmoidweighting)
        mask_ws = construct_wsmask(outpaths[dsname], props, None,
                                   h5path_mds, h5path_mmm,
                                   MAdilation=[], dist_thr=0.0)
        # TODO: remove MA seeds from mask
        label_mm = separate_sheaths_sigmoid(outpaths, props, ma, seeds,
                                            label_mm,
                                            mask_ws, sigmoidweighting, margin,
                                            medwidth_file)

    label_mm.close()
    mask_ws.close()
    ma.close()


def write_dist(outpath, props, dist_data):

    props['dtype'] = 'float'
    dist = Image(outpath, **props)
    dist.create()
    dist.write(dist_data)

    return dist


def ws(outpath, props, mask_ma, seeds, mask_ws, dist):

    props['dtype'] = 'uint64'
    label_mm = LabelImage(outpath, **props)
    label_mm.create()
    ws = watershed(img_as_float(dist.ds[:]), seeds, mask=mask_ws.ds[:])
    ws[mask_ma] = 0
    label_mm.write(ws)

    return label_mm


def separate_sheaths_simple(outpaths, props, mask_ma, seeds, mask_ws, dist_thr=0.25):

    abs_es = np.absolute(props['elsize'])

    props['dtype'] = 'float'
    dist = Image(outpaths['distance'], **props)
    dist.create()
    dist.write(distance_transform_edt(~mask_ma, sampling=abs_es))

    props['dtype'] = 'uint64'
    label_mm = LabelImage(outpaths['sheaths'], **props)
    label_mm.create()
    ws = watershed(img_as_float(dist.ds[:]), seeds, mask=mask_ws.ds[:])
    ws[mask_ma] = 0
    label_mm.write(ws)

    dist.close()
    label_mm.close()


def separate_sheaths_sigmoid(outpaths, props, ma, seeds, label_mm_init, mask_ws, weight, margin, medwidth_file=''):

    dsname = 'distance_sigmoid_{}'.format(weight)
    dist_data, dist_mask, medwidths = distance_transform_sw(ma,
                                                            label_mm_init,
                                                            weight, margin,
                                                            medwidth_file)
    dist = write_dist(outpaths[dsname], props, dist_data)

    dsname = 'wsmask_sigmoid_{}'.format(weight)
    mask_ws_data = np.logical_and(mask_ws.ds[:], dist_mask)
    props['dtype'] = 'bool'
    mask_ws = MaskImage(outpaths[dsname], **props)
    mask_ws.create()
    mask_ws.write(mask_ws_data)

    comps = ma.split_path(outpaths['out'])
    write_medwidths(comps, medwidths)

    dsname = 'sheaths_sigmoid_{}'.format(weight)
    mask_ma = ma.ds[:] != 0
    label_mm = ws(outpaths[dsname], props, mask_ma, seeds, mask_ws, dist)

    dist.close()

    return label_mm


def write_medwidths(comps, medwidths):

    filepath = '{}_{}.pickle'.format(comps['base'], comps['dset'])
    with open(filepath, "wb") as f:
        pickle.dump(medwidths, f)

    filepath = '{}_{}.txt'.format(comps['base'], comps['dset'])
    with open(filepath, "wb") as f:
        for lsk, lsv in medwidths.items():
            f.write("%8d: " % lsk)
            f.write("%8f " % lsv)
            f.write('\n')


def get_outpaths(h5path_out, save_steps, weight):

    outpaths = {'out': h5path_out,
                'wsmask': '',
                'distance': '', 'sheaths': ''}
    if weight:
        outpaths['wsmask_sigmoid_{}'.format(weight)] = ''
        outpaths['distance_sigmoid_{}'.format(weight)] = ''
        outpaths['sheaths_sigmoid_{}'.format(weight)] = ''
    if save_steps:
        outpaths = utils.gen_steps(outpaths, save_steps)

    return outpaths


def construct_wsmask(outpath, props, mask_ma,
                     h5path_mds='', h5path_mmm='',
                     MAdilation=[1, 7, 7], dist_thr=0.0):
    """Construct a mask of valid myelin voxels."""

    mask = np.ones(props['shape'], dtype='bool')

    # create mask covering the myelinated axon compartment
    if dist_thr:  # threshold the distance image
        mask = mask_ma.ds[:] < dist_thr
        print("thresholded distance image at {}; {} voxels in mask".format(dist_thr, np.sum(mask[:])))
    elif MAdilation:  # dilate seedmask and remove voxels in seedmask
        if len(MAdilation) == 1:
            binary_dilation(mask_ma, iterations=MAdilation, output=mask)
        elif len(MAdilation) == 3:
            struct = generate_anisotropic_selem(MAdilation)
            binary_dilation(mask_ma, structure=struct, iterations=1, output=mask)
        np.logical_xor(mask, mask_ma, mask)
        print("dilated mask_ma with {}; {} voxels in mask".format(MAdilation, np.sum(mask[:])))

    # mask with myelin mask
    if h5path_mmm:
        mask_mm = utils.get_image(h5path_mmm)
        np.logical_and(mask, mask_mm.ds[:].astype('bool'), mask)
        print("masked with {}; {} voxels in mask".format(h5path_mmm, np.sum(mask[:])))

    # mask with dataset mask
    if h5path_mds:
        mask_ds = utils.get_image(h5path_mds)
        np.logical_and(mask, mask_ds.ds[:].astype('bool'), mask)
        print("masked with {}; {} voxels in mask".format(h5path_mds, np.sum(mask[:])))

    props['dtype'] = 'bool'
    mask_ws = MaskImage(outpath, **props)
    mask_ws.create()
    mask_ws.write(data=mask)

    return mask_ws


def distance_transform_sw(labelMA, labelMM, weight=1, margin=50, medwidth_file=''):
    """Calculate the sum of sigmoid-weighted distance transforms."""

    elsize = np.absolute(labelMA.elsize)

    # load median sheath widths if provided
    if medwidth_file:
        with open(medwidth_file, 'rb') as f:
            medwidths = pickle.load(f)
    else:
        medwidths = {}

    distsum = np.ones_like(labelMM.ds[:], dtype='float')
    mask = np.zeros_like(labelMM.ds[:], dtype='bool')

    rp = regionprops(labelMA.ds[:])
    for prop in rp:

        # get data cutout labels with margin
        x, X, y, Y, z, Z = get_coords(prop, margin, labelMM.ds.shape)
        MA_region = labelMA.ds[z:Z, y:Y, x:X]
        MM_region = labelMM.ds[z:Z, y:Y, x:X]
        distsum_region = distsum[z:Z, y:Y, x:X]
        mask_region = mask[z:Z, y:Y, x:X]

        # get label distance map
        maskMA = MA_region == prop.label
        dist = distance_transform_edt(~maskMA, sampling=elsize)

        # calculate the median width of the label
        if prop.label not in medwidths.keys():
            rim = get_rim(prop, MA_region, MM_region)
            medwidths[prop.label] = np.median(dist[rim])
        mw = medwidths[prop.label]
        print("median width for {}: {}".format(prop.label, mw))

        # update the mask
        dist_mask = dist < mw * 1.5  # TODO: make argument
        np.logical_or(mask_region, dist_mask, mask_region)
        mask[z:Z, y:Y, x:X] = mask_region

        # add the weighted distance map
        wdist = 2 * (expit(dist / (weight * mw)) - 0.5)
#         w = weight * mw
#         dist *= 1/w
#         wdist = 2 * (expit(x) - 0.5)
        distsum_region = np.minimum(distsum_region, wdist)
        distsum[z:Z, y:Y, x:X] = distsum_region

    return distsum, mask, medwidths


def get_coords(prop, margin, dims):

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

    return x, X, y, Y, z, Z


def get_rim(prop, MA_region, MM_region):

    MFregion = MA_region + MM_region
    maskMF = MFregion == prop.label
    rim = maskMF - binary_erosion(maskMF, border_value=1)

    return rim


if __name__ == "__main__":
    main(sys.argv)
