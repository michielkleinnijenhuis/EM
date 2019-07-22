#!/usr/bin/env python

"""Fill holes in labels.

"""

import sys
import argparse

import numpy as np
from scipy.ndimage.morphology import (binary_dilation,
                                      binary_erosion,
                                      binary_closing,
                                      binary_fill_holes,
                                      grey_dilation)
from skimage.measure import label, regionprops
from skimage.morphology import watershed

from wmem import parse, utils, wmeMPI, Image, MaskImage, LabelImage
from wmem.merge_slicelabels import generate_anisotropic_selem
from wmem.merge_labels import get_region_slices_around


def main(argv):
    """Fill holes in labels."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_fill_holes(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    fill_holes(
        args.inputfile,
        args.methods,
        args.selem,
        args.labelmask,
        args.maskDS,
        args.maskMM,
        args.maskMX,
        args.outputfile,
        args.outputholes,
        args.outputMA,
        args.outputMM,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def fill_holes(
        image_in,
        methods,
        selem,
        h5path_mask='',
        h5path_md='',
        h5path_mm='',
        h5path_mx='',
        outputpath='',
        h5path_out_holes='',
        h5path_out_ma='',
        h5path_out_mm='',
        protective=False,
        usempi=False,
        ):
    """Fill holes in labels."""

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, comm=mpi.comm)
    if h5path_mask:
        mask = utils.get_image(h5path_mask, comm=mpi.comm)
        # TODO: mask im
#         ds_in[~ds_ml[:].astype('bool')] = 0
        mask.close()

    # Determine the properties of the output dataset.
    props = im.get_props(protective=protective, squeeze=True)
    mo = LabelImage(outputpath, **props)
    mo.create(comm=mpi.comm)

    # fill holes
    if mpi.rank == 0:
        mo.write(im.ds)

    for m in methods:
        if m == '5':
            fill_holes_m5(mo, selem, mpi)
        else:
            if m == '1':
                mo.ds[:] = fill_holes_m1(mo.ds[:], selem)
            if m == '2':
                mo.ds[:] = fill_holes_m2(mo.ds[:], selem)
            if m == '3':
                mo.ds[:] = fill_holes_m3(mo.ds[:], selem)
            if m == '4':
                mo.ds[:] = fill_holes_m4(mo.ds[:], h5path_md, h5path_mm, h5path_mx)
            mo.write()

#     # additional output: holes filled and updated masks
#     if h5path_out_holes:
#         h5file_ho, ds_ho, _, _ = utils.h5_write(None,
#                                                 ds_out.shape, ds_out.dtype,
#                                                 h5path_out_holes,
#                                                 element_size_um=elsize,
#                                                 axislabels=axlab)
#         ds_ho[:] = ds_out[:]
#         ds_ho[ds_in > 0] = 0
#         h5file_ho.close()
#     if h5path_out_ma:
#         ds_ma = utils.h5_write(ds_out.astype('bool'), ds_out.shape, 'uint8',
#                                h5path_out_ma,
#                                element_size_um=elsize,
#                                axislabels=axlab)[1]
#     if h5path_out_mm:
#         ds_mm = utils.h5_load(h5path_mm, load_data=True)
#         ds_mm[ds_ho > 0] = 0
#         ds_mm = utils.h5_write(ds_mm.astype('bool'), ds_mm.shape, 'uint8',
#                                h5path_out_mm,
#                                element_size_um=elsize,
#                                axislabels=axlab)

    im.close()
    mo.close()

    return mo


# ========================================================================== #
# function defs
# ========================================================================== #


def fill_holes_m1(labels, selem=[3, 3, 3]):
    """Fill holes in labels."""

    binim = labels != 0
    # does binary_closing bridge seperate labels? YES, and eats from boundary
    # binim = binary_closing(binim, iterations=10)
    holes = label(~binim, connectivity=1)

    labelCount = np.bincount(holes.ravel())
    background = np.argmax(labelCount)
    holes[holes == background] = 0

    labels_dil = grey_dilation(labels, size=selem)

    rp = regionprops(holes, labels_dil)
    mi = {prop.label: prop.max_intensity for prop in rp}
    fw = [mi[key] if key in mi.keys() else 0
          for key in range(0, np.amax(holes) + 1)]
    fw = np.array(fw)

    holes_remapped = fw[holes]

    labels = np.add(labels, holes_remapped)

    return labels


def fill_holes_m2(labels, selem=[3, 3, 3]):
    """Fill holes in labels."""

    rp = regionprops(labels)
    for prop in rp:
        print(prop.label)
        z, y, x, Z, Y, X = tuple(prop.bbox)
        mask = prop.image
#             mask = binary_fill_holes(mask)
        mask = binary_closing(mask, iterations=selem[0])
        mask = binary_fill_holes(mask)
        imregion = labels[z:Z, y:Y, x:X]
        imregion[mask] = prop.label

    return labels


def fill_holes_m5(labels, selem=[1, 7, 7], mpi=None):
    """Fill holes in labels."""

    struct = generate_anisotropic_selem(selem)
    rp = regionprops(labels.ds)

    # Prepare for processing with MPI.
    mpi.nblocks = len(rp)
    mpi.scatter_series()  # randomize=True

#     for prop in rp:
    for i in mpi.series:
        prop = rp[i]

        labels.slices = get_region_slices_around(labels, prop, selem)[0]
        print(prop.label, labels.slices)
        imregion = labels.slice_dataset()
        mask = imregion == prop.label

        z, Z = labels.slices[0].start, labels.slices[0].stop

        if z == 0:
            mask = np.insert(mask, 0, True, axis=0)
        if Z == labels.ds.shape[0]:
            mask = np.insert(mask, mask.shape[0], True, axis=0)

        mask = binary_closing(mask, structure=struct, iterations=selem[0])
        mask = binary_fill_holes(mask)

        if Z == labels.ds.shape[0]:
            mask = mask[:-1, :, :]
        if z == 0:
            mask = mask[1:, :, :]

        imregion[mask] = prop.label

        labels.write(imregion)

#     return labels


def fill_holes_m3(labels, selem=[3, 3, 3]):
    """Fill holes in labels."""

    rp = regionprops(labels)
    for prop in rp:

        z, y, x, Z, Y, X = tuple(prop.bbox)

        mask = prop.image

        if z == 0:
            mask = np.insert(mask, 0, True, axis=0)
        if Z == labels.shape[0]:
            mask = np.insert(mask, mask.shape[0], True, axis=0)

        mask = binary_fill_holes(mask)

        if Z == labels.shape[0]:
            mask = mask[:-1, :, :]
        if z == 0:
            mask = mask[1:, :, :]

        imregion = labels[z:Z, y:Y, x:X]
        imregion[mask] = prop.label

    return labels


def fill_holes_m4(labels, h5path_md, h5path_mm, h5path_mx):
    """Fill holes in labels."""

    ds_md = utils.h5_load(h5path_md, load_data=True)[1]
    ds_mm = utils.h5_load(h5path_mm, load_data=True)[1]
    mask = ~ds_md | ds_mm
    MMlabels = fill_holes_watershed(labels, mask)

    ds_mx = utils.h5_load(h5path_mx, load_data=True)[1]
    mask = ~ds_md | ds_mx
    MXlabels = fill_holes_watershed(labels, mask)

    labels = np.maximum(MMlabels, MXlabels)

    return labels


def fill_holes_watershed(labels, mask_in):
    """Fill holes not reachable from unmyelinated axons space."""

    mask = mask_in | labels.astype('bool')
    labels_mask = label(~mask)

    counts = np.bincount(labels_mask.ravel())
    bg = np.argmax(counts[1:]) + 1

    mask = mask_in | (labels_mask == bg)
    labels = watershed(mask, labels, mask=~mask)

    return labels


def get_mask(h5path_in1, h5path_in2):
    """Load the set of masks for method4."""

    h5file_in1, ds_in1, _, _ = utils.h5_load(h5path_in1, load_data=True)
    h5file_in2, ds_in2, _, _ = utils.h5_load(h5path_in2, load_data=True)
    mask = ~ds_in1 | ds_in2

    return mask


if __name__ == "__main__":
    main(sys.argv[1:])
