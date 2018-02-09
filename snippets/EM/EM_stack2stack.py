#!/usr/bin/env python

import os
import sys
import argparse

import h5py

from nibabel import Nifti1Image

from skimage.io import imsave
from skimage import img_as_ubyte
from skimage.transform import downscale_local_mean


def main(argv):

    parser = argparse.ArgumentParser(description='Juggle around stacks.')
    parser.add_argument('inputfile', help='the inputfile')
    parser.add_argument('outputfile', help='the outputfile')
    parser.add_argument('-f', '--fieldnamein',
                        help='input hdf5 fieldname <stack>')
    parser.add_argument('-g', '--fieldnameout',
                        help='output hdf5 fieldname <stack>')
    parser.add_argument('-d', '--datatype',
                        help='the numpy-style output datatype')
    parser.add_argument('-i', '--inlayout',
                        help='the data layout of the input')
    parser.add_argument('-l', '--outlayout',
                        help='the data layout for output')
    parser.add_argument('-s', '--chunksize', type=int, nargs='*',
                        help='hdf5 chunk sizes (in order of outlayout)')
    parser.add_argument('-e', '--element_size_um', type=float, nargs='*',
                        help='dataset element sizes (in order of outlayout)')
    parser.add_argument('-n', '--nzfills', type=int, default=5,
                        help='number of characters at the end that define z')
    parser.add_argument('-p', '--dset_name', default=None,
                        help='the identifier of the datablock')
    parser.add_argument('-b', '--blockoffset', nargs=3, type=int, default=[0, 0, 0],
                        help='...')
    parser.add_argument('-m', '--enable_multi_output', nargs='*', default=[],
                        help='output the stack in jpg,png,tif,nii,h5')
    parser.add_argument('-r', '--downscale', type=int, nargs='*', default=[],
                        help='factors to downscale (in order of outlayout)')
    parser.add_argument('-u', '--uint8conv', action='store_true',
                        help='convert data to uint8')
    parser.add_argument('-x', default=0, type=int, help='first x-index')
    parser.add_argument('-X', type=int, help='last x-index')
    parser.add_argument('-y', default=0, type=int, help='first y-index')
    parser.add_argument('-Y', type=int, help='last y-index')
    parser.add_argument('-z', default=0, type=int, help='first z-index')
    parser.add_argument('-Z', type=int, help='last z-index')
    parser.add_argument('-c', default=0, type=int, help='first c-index')
    parser.add_argument('-C', type=int, help='last C-index')
    parser.add_argument('-t', default=0, type=int, help='first t-index')
    parser.add_argument('-T', type=int, help='last T-index')
    args = parser.parse_args()

    inputfile = args.inputfile
    outputfile = args.outputfile
    dset_name = args.dset_name
    blockoffset = args.blockoffset

    uint8conv = args.uint8conv

    f = h5py.File(inputfile, 'r')

    if args.fieldnamein:
        infield = args.fieldnamein
    else:
        grps = [name for name in f]
        infield = grps[0]

    if args.fieldnameout:
        outfield = args.fieldnameout
    elif inputfile != outputfile:
        outfield = infield
    else:
        outfield = 'stack'

    inds = f[infield]
    inshape = inds.shape
    indim = len(inds.dims)  # inds.ndim

    if 'dims' in inds.attrs.keys():
        inlayout = [d.label for d in inds.dims]
    elif args.inlayout:
        inlayout = args.inlayout
    else:
        inlayout = 'xyzct'[0:indim]

    if args.outlayout:
        outlayout = args.outlayout
    elif inlayout:
        outlayout = inlayout
    else:
        outlayout = 'xyzct'[0:indim]

    std2in = ['xyzct'.index(l) for l in inlayout]
    std2out = ['xyzct'.index(l) for l in outlayout]
    in2out = [inlayout.index(l) for l in outlayout]

    if args.element_size_um:
        element_size_um = args.element_size_um
    elif 'element_size_um' in inds.attrs.keys():
        element_size_um = [inds.attrs['element_size_um'][i]
                           for i in in2out]
    else:
        element_size_um = None

    if args.chunksize:
        chunksize = args.chunksize
    elif inds.chunks:
        chunksize = [inds.chunks[i] for i in in2out]

    datatype = args.datatype if args.datatype else inds.dtype

    if dset_name is not None:
        _, x, X, y, Y, z, Z = split_filename(outputfile, blockoffset)
    else:
        x = args.x
        if args.X:
            X = args.X
        elif 'x' in inlayout:
            X = inshape[inlayout.index('x')]
        else:
            X = None
        y = args.y
        if args.Y:
            Y = args.Y
        elif 'y' in inlayout:
            Y = inshape[inlayout.index('y')]
        else:
            Y = None
        z = args.z
        if args.Z:
            Z = args.Z
        elif 'z' in inlayout:
            Z = inshape[inlayout.index('z')]
        else:
            Z = None
    c = args.c
    if args.C:
        C = args.C
    elif 'c' in inlayout:
        C = inshape[inlayout.index('c')]
    else:
        C = None
    t = args.t
    if args.T:
        T = args.T
    elif 't' in inlayout:
        T = inshape[inlayout.index('t')]
    else:
        T = None

    stdsel = [[x, X], [y, Y], [z, Z], [c, C], [t, T]]
    insel = [stdsel[i] for i in std2in]

    downscale = tuple(args.downscale)

    b, e = os.path.splitext(outputfile)
    if e == '.gz':
        b, e2 = os.path.splitext(b)  # catch double extension .nii.gz
        e = e2 + e
    outexts = args.enable_multi_output
    outexts.append(e)
    outexts = list(set(outexts))
    nzfills = args.nzfills
#     if b[-nzfills:].isdigit():
#         slcoffset = int(b[-nzfills:])
#         b = b[:-nzfills]
#     else:
#         slcoffset = 0

    # TODO: most memory-efficient solution?
    if indim == 2:
        img = inds[insel[0][0]:insel[0][1],
                   insel[1][0]:insel[1][1]]
    elif indim == 3:
        img = inds[insel[0][0]:insel[0][1],
                   insel[1][0]:insel[1][1],
                   insel[2][0]:insel[2][1]]
    elif indim == 4:
        img = inds[insel[0][0]:insel[0][1],
                   insel[1][0]:insel[1][1],
                   insel[2][0]:insel[2][1],
                   insel[3][0]:insel[3][1]]
    elif indim == 5:
        img = inds[insel[0][0]:insel[0][1],
                   insel[1][0]:insel[1][1],
                   insel[2][0]:insel[2][1],
                   insel[3][0]:insel[3][1],
                   insel[4][0]:insel[4][1]]
    f.close()

    if outlayout != inlayout:
        img = img.transpose(in2out)
    if downscale:
        img = downscale_local_mean(img, downscale)  # FIXME: big stack will encounter memory limitations here
        if element_size_um is not None:
            element_size_um = [el * downscale[i]
                               for i, el in enumerate(element_size_um)]
    if datatype != img.dtype:
        img = img.astype(datatype, copy=False)
    if uint8conv:
        img = img_as_ubyte(img)

    for ext in outexts:
        if '.nii' in ext:  # TODO?: nifti is always xyztc?
            mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            if element_size_um is not None:
                mat[0][0] = element_size_um[0]
                mat[1][1] = element_size_um[1]
                mat[2][2] = element_size_um[2]
            Nifti1Image(img, mat).to_filename(b + '.nii.gz')

        if '.h5' in ext:
            # FIXME: somehow 'a' doesn't work if file doesnt exist
            otype = 'w'  #'a' if path.isfile(outputfile) else 'w'
            g = h5py.File(b + '.h5', otype)
#             datalayout = tuple(stdsel[i][1]-stdsel[i][0] for i in std2out)
            if 'chunksize' in locals():
                outds = g.create_dataset(outfield, img.shape,
                                         chunks=tuple(chunksize),
                                         dtype=datatype, compression='gzip')
            else:
                outds = g.create_dataset(outfield, img.shape,
                                         dtype=datatype, compression='gzip')
            outds[:] = img
            if element_size_um is not None:
                outds.attrs['element_size_um'] = element_size_um
            for i, l in enumerate(outlayout):
                outds.dims[i].label = l
            g.close()

        if (('.tif' in ext) |
            ('.png' in ext) |
            ('.jpg' in ext)) & (indim == 3):  # only 3D for now
            if not os.path.exists(b):
                os.makedirs(b)
            for slc in range(0, img.shape[outlayout.index('z')]):
                slcno = slc  # + slcoffset
                if outlayout.index('z') == 0:
                    slcdata = img[slc, :, :]
                elif outlayout.index('z') == 1:
                    slcdata = img[:, slc, :]
                elif outlayout.index('z') == 2:
                    slcdata = img[:, :, slc]
                imsave(os.path.join(b, str(slcno).zfill(nzfills) + ext), slcdata)


# ========================================================================== #
# function defs
# ========================================================================== #


def dataset_name(dset_info):
    """Return the basename of the dataset."""

    nf = dset_info['nzfills']
    dname = dset_info['base'] + \
                '_' + str(dset_info['x']).zfill(nf) + \
                '-' + str(dset_info['X']).zfill(nf) + \
                '_' + str(dset_info['y']).zfill(nf) + \
                '-' + str(dset_info['Y']).zfill(nf) + \
                '_' + str(dset_info['z']).zfill(nf) + \
                '-' + str(dset_info['Z']).zfill(nf) + \
                dset_info['postfix']

    return dname


def split_filename(filename, blockoffset=[0, 0, 0]):
    """Extract the data indices from the filename."""

    datadir, tail = os.path.split(filename)
    fname = os.path.splitext(tail)[0]
    parts = fname.split("_")
    x = int(parts[1].split("-")[0]) - blockoffset[0]
    X = int(parts[1].split("-")[1]) - blockoffset[0]
    y = int(parts[2].split("-")[0]) - blockoffset[1]
    Y = int(parts[2].split("-")[1]) - blockoffset[1]
    z = int(parts[3].split("-")[0]) - blockoffset[2]
    Z = int(parts[3].split("-")[1]) - blockoffset[2]

    dset_info = {'datadir': datadir, 'base': parts[0],
                 'nzfills': len(parts[1].split("-")[0]),
                 'postfix': '_'.join(parts[4:]),
                 'x': x, 'X': X, 'y': y, 'Y': Y, 'z': z, 'Z': Z}

    return dset_info, x, X, y, Y, z, Z


if __name__ == "__main__":
    main(sys.argv)
