#!/usr/bin/env python

"""Create thresholded hard segmentations.

"""

import sys
import argparse

import numpy as np

from skimage.morphology import remove_small_objects
from skimage.morphology import binary_dilation, ball, disk

from wmem import parse, utils


def main(argv):
    """Create thresholded hard segmentations."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_prob2mask(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    if args.step:
        for thr in frange(args.step, 1.0, args.step):
            prob2mask(
                args.inputfile,
                args.dataslices,
                thr,
                args.upper_threshold,
                args.size,
                args.dilation,
                args.inputmask,
                args.outputfile.format(thr),
                args.save_steps,
                args.protective,
                )
    else:
        prob2mask(
            args.inputfile,
            args.dataslices,
            args.lower_threshold,
            args.upper_threshold,
            args.size,
            args.dilation,
            args.inputmask,
            args.outputfile,
            args.save_steps,
            args.protective,
            )


def prob2mask(
        h5path_in,
        dataslices=None,
        lower_threshold=0,
        upper_threshold=1,
        size=0,
        dilation=0,
        h5path_mask=None,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Create thresholded hard segmentation."""

    # check output paths
    outpaths = {'out': h5path_out, 'raw': '', 'mito': '', 'dil': ''}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return
#     if bool(h5path_out) and ('.h5' in h5path_out):
#         status, info = utils.h5_check(h5path_out, protective)
#         print(info)
#         if status == "CANCELLED":
#             return

    # open data for reading
    h5file_in, ds_in, es, al = utils.load(h5path_in)

    slices = utils.get_slice_objects(dataslices, ds_in.shape)
    datashape_out = (len(range(*slices[0].indices(slices[0].stop))),
                     len(range(*slices[1].indices(slices[1].stop))),
                     len(range(*slices[2].indices(slices[2].stop))))

    data, _, _, slices_out = utils.load_dataset(
        ds_in, es, al, al, dtype='', dataslices=dataslices)

    if h5path_mask is not None:
        inmask = utils.load(h5path_mask, load_data=True, dtype='bool',
                            dataslices=dataslices)[0]

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, datashape_out, 'uint8',
                                        outpaths['out'],
                                        element_size_um=es,
                                        axislabels=al)
    if save_steps:
        h5file_mr, ds_mr = utils.h5_write(None, datashape_out, 'uint8',
                                          outpaths['raw'],
                                          element_size_um=es,
                                          axislabels=al)
        h5file_mm, ds_mm = utils.h5_write(None, datashape_out, 'uint8',
                                          outpaths['mito'],
                                          element_size_um=es,
                                          axislabels=al)
        h5file_md, ds_md = utils.h5_write(None, datashape_out, 'uint8',
                                          outpaths['dil'],
                                          element_size_um=es,
                                          axislabels=al)

    go2D = True
    if go2D:
        mask = np.zeros_like(data)
        mr = np.zeros_like(data)
        mm = np.zeros_like(data)
        md = np.zeros_like(data)
        for i, slc in enumerate(data):
            smf, smr, smm, smd = process_slice(slc,
                                               lower_threshold,
                                               upper_threshold,
                                               size, dilation, se=disk)
            mask[i, :, :] = smf
            mr[i, :, :] = smr
            mm[i, :, :] = smm
            md[i, :, :] = smd
    else:
        mask, mr, mm, md = process_slice(data,
                                         lower_threshold,
                                         upper_threshold,
                                         size, dilation, se=ball)

    if h5path_mask is not None:
        mask[~inmask] = False

    if '.h5' in h5path_out:
        utils.write_to_h5ds(ds_out, mask.astype('uint8'), slices_out)
        if save_steps:
            utils.write_to_h5ds(ds_mr, mr.astype('uint8'), slices_out)
            utils.write_to_h5ds(ds_mm, mm.astype('uint8'), slices_out)
            utils.write_to_h5ds(ds_md, md.astype('uint8'), slices_out)
    elif h5path_out.endswith('.tif'):
        utils.imf_write(h5path_out, mask.astype('uint8'))
        # TODO: save_steps?

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
        if save_steps:
            h5file_mr.close()
            h5file_mm.close()
            h5file_md.close()
    except (ValueError, AttributeError):
        return mask, mr, mm, md


def process_slice(data, lt, ut, size=0, dilation=0, se=disk):
    """Threshold and filter data."""

    mask_mito, mask_dil = None, None

    mask_raw = np.logical_and(data > lt, data <= ut)
    mask = mask_raw

    if dilation:
        mask_dil = binary_dilation(mask, selem=se(dilation))
        mask = mask_dil

    if size:
        mask_filter = remove_small_objects(mask, min_size=size)
        mask_mito = np.logical_xor(mask, mask_filter)
        mask = mask_filter

    return mask, mask_raw, mask_mito, mask_dil


def frange(x, y, jump):
    """Range for floats."""
    while x < y:
        yield x
        x += jump


if __name__ == "__main__":
    main(sys.argv[1:])
