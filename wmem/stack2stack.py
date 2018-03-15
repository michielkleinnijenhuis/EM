#!/usr/bin/env python

"""Convert/select/downscale/transpose/... an hdf5 dataset.

"""

import sys
import argparse
import os

from wmem import parse, utils


def main(argv):
    """Convert/select/downscale/transpose/... an hdf5 dataset."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_stack2stack(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    stack2stack(
        args.inputfile,
        args.outputfile,
        args.dset_name,
        args.blockoffset,
        args.datatype,
        args.uint8conv,
        args.inlayout,
        args.outlayout,
        args.element_size_um,
        args.chunksize,
        args.additional_outputs,
        args.nzfills,
        args.dataslices,
        args.save_steps,
        args.protective,
        )


def stack2stack(
        inputfile,
        outputfile,
        dset_name='',
        blockoffset=[],
        datatype='',
        uint8conv=False,
        inlayout='',
        outlayout='',
        elsize=[],
        chunksize=[],
        additional_outputs=[],
        nzfills=5,
        dataslices=None,
        save_steps=False,
        protective=False,
        ):
    """Convert/select/downscale/transpose/... an hdf5 dataset."""

    # output root and exts
    root, ext = split_filepath(outputfile)
    outexts = list(set(additional_outputs + [ext]))

    # Check if any output paths already exist.
    outpaths = {'out': outputfile, 'addext': (root, additional_outputs)}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    h5file_in, ds_in, h5elsize, h5axlab = utils.h5_load(inputfile)

    try:
        ndim = ds_in.ndim
    except AttributeError:
        ndim = len(ds_in.dims)

    # data layout
        # FIXME: h5axlab not necessarily xyzct!
        # FIXME: this forces a possibly erroneous outlayout when inlayout is not found
    inlayout = inlayout or ''.join(h5axlab) or 'zyxct'[0:ndim]
    outlayout = outlayout or inlayout
    in2out = [inlayout.index(l) for l in outlayout]

    # element size
    elsize = elsize or (h5elsize[in2out]
                        if h5elsize is not None else None)

    # chunksize
    if chunksize is not None:
        chunksize = tuple(chunksize) or (
            True if not any(chunksize)
            else (tuple([ds_in.chunks[i] for i in in2out])
                  if ds_in.chunks else None
                  )
            )

    # datatype
    datatype = datatype or ds_in.dtype

    if dset_name:
        _, x, X, y, Y, z, Z = utils.split_filename(dset_name, blockoffset)
        slices = {'x': [x, X, 1], 'y': [y, Y, 1], 'z': [z, Z, 1]}
        if ndim > 3:
            C = ds_in.shape[inlayout.index('c')]
            slices['c'] = [0, C, 1]
        sliceslist = [slices[dim] for dim in inlayout]
        dataslices = [item for sl in sliceslist for item in sl]

    # get the selected and transformed data
    # TODO: most memory-efficient solution
    data = utils.load_dataset(ds_in, elsize, inlayout, outlayout,
                              datatype, dataslices, uint8conv)[0]

    h5file_in.close()

    # write the data
    for ext in outexts:

        if '.nii' in ext:
            if data.dtype == 'float16':
                data = data.astype('float')
            utils.write_to_nifti(root + '.nii.gz', data, elsize)

        if '.h5' in ext:
            utils.h5_write(data, data.shape, data.dtype,
                           outputfile,
                           element_size_um=elsize,
                           axislabels=outlayout,
                           chunks=chunksize)

        if (('.tif' in ext) |
                ('.png' in ext) |
                ('.jpg' in ext)) & (data.ndim < 4):
            if data.ndim == 2:
                data = data.atleast_3d()
                outlayout += 'z'
            utils.write_to_img(root, data, outlayout, nzfills, ext)

    return data


def split_filepath(outputpath):
    """Split a filepath in root and extension."""

    if '.h5' in outputpath:  # ext in middle of outputfilepath
        ext = '.h5'
        root = outputpath.split(ext)[0]
    elif outputpath.endswith('.nii.gz'):  # double ext
        ext = '.nii.gz'
        root = outputpath.split(ext)[0]
    else:  # simple ext
        root, ext = os.path.splitext(outputpath)

    return root, ext


if __name__ == "__main__":
    main(sys.argv)
