#!/usr/bin/env python

"""Convert/select/downscale/transpose/... an hdf5 dataset.

"""

import sys
import argparse
import os

from skimage import img_as_ubyte
from skimage.transform import downscale_local_mean

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
        args.datatype,
        args.blockoffset,
        args.uint8conv,
        args.inlayout,
        args.outlayout,
        args.element_size_um,
        args.chunksize,
        args.downscale,
        args.additional_outputs,
        args.nzfills,
        (args.x, args.X,
         args.y, args.Y,
         args.z, args.Z,
         args.c, args.C,
         args.t, args.T),
#         args.dataslices,
        args.protective,
        )


def stack2stack(
        inputfile,
        outputfile,
        dset_name='',
        datatype='',
        blockoffset=[],
        uint8conv=False,
        inlayout='',
        outlayout='',
        elsize=[],
        chunksize=[],
        downscale=[],
        additional_outputs=[],
        nzfills=5,
        xyzct=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#         dataslices=None,
        protective=False,
        ):
    """Convert/select/downscale/transpose/... an hdf5 dataset."""

    # TODO: protect other file formats
    if '.h5' in outputfile:
        status, info = utils.h5_check(outputfile, protective)
        print(info)
        if status == "CANCELLED":
            return

    h5file_in, ds_in, h5elsize, h5axlab = utils.h5_load(inputfile)
    try:
        ndim = ds_in.ndim
    except AttributeError:
        ndim = len(ds_in.dims)

    # output root and exts
    if '.h5' in outputfile + ''.join(additional_outputs):
        root = outputfile.split('.h5')[0]
        ext = '.h5'
    elif outputfile.endswith('.nii.gz'):
        root = os.path.splitext(outputfile)[0]
        root = os.path.splitext(root)[0]
        ext = '.nii.gz'
    else:
        root, ext = os.path.splitext(outputfile)
    outexts = list(set(additional_outputs + [ext]))

    # data layout  # FIXME: h5axlab not necessarily xyzct!
    inlayout = inlayout or ''.join(h5axlab) or 'zyxct'[0:ndim]
    outlayout = outlayout or inlayout
    in2out = [inlayout.index(l) for l in outlayout]

    # element size
    elsize = elsize or (h5elsize[in2out] if h5elsize is not None else None)

    # chunksize
    if chunksize is not None:
        chunksize = tuple(chunksize) or (
            True if not any(chunksize)
            else (tuple([ds_in.chunks[i] for i in in2out]) if ds_in.chunks
                  else None
                  )
            )

    # datatype
    datatype = datatype or ds_in.dtype

    # subset range
    x, X, y, Y, z, Z, c, C, t, T = xyzct

    def get_bound(dim):
        return ds_in.shape[inlayout.index(dim)] if dim in inlayout else None

    if dset_name:  # override from filename FIXME? dset_name / outputfile
        _, x, X, y, Y, z, Z = utils.split_filename(outputfile, blockoffset)
    else:
        X = X or get_bound('x')
        Y = Y or get_bound('y')
        Z = Z or get_bound('z')
    C = C or get_bound('c')
    T = T or get_bound('t')

    stdsel = [[x, X], [y, Y], [z, Z], [c, C], [t, T]]
    std2in = ['xyzct'.index(l) for l in inlayout]
    insel = [stdsel[i] for i in std2in]

    # get the data  # TODO: most memory-efficient solution
    if ndim == 2:
        data = ds_in[insel[0][0]:insel[0][1],
                     insel[1][0]:insel[1][1]]
    elif ndim == 3:
        data = ds_in[insel[0][0]:insel[0][1],
                     insel[1][0]:insel[1][1],
                     insel[2][0]:insel[2][1]]
    elif ndim == 4:
        data = ds_in[insel[0][0]:insel[0][1],
                     insel[1][0]:insel[1][1],
                     insel[2][0]:insel[2][1],
                     insel[3][0]:insel[3][1]]
    elif ndim == 5:
        data = ds_in[insel[0][0]:insel[0][1],
                     insel[1][0]:insel[1][1],
                     insel[2][0]:insel[2][1],
                     insel[3][0]:insel[3][1],
                     insel[4][0]:insel[4][1]]

    h5file_in.close()

    # do any data conversions
    if outlayout != inlayout:
        data = data.transpose(in2out)
    if downscale:
        data = downscale_local_mean(data, tuple(downscale))
        # FIXME: big stack will encounter memory limitations here
        if elsize is not None:
            elsize = [el * downscale[i] for i, el in enumerate(elsize)]
    if uint8conv:
        data = img_as_ubyte(data)
    elif datatype != data.dtype:
        data = data.astype(datatype, copy=False)

    # write the data
    for ext in outexts:
        if '.nii' in ext:
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


if __name__ == "__main__":
    main(sys.argv)
