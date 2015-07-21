#!/usr/bin/env python

import sys
import argparse
from os import path
from nibabel import Nifti1Image
import h5py

def main(argv):
    
    parser = argparse.ArgumentParser(description=
        'Juggle around with hdf5 stacks.')
    
    parser.add_argument('inputfile', help='the inputfile')
    parser.add_argument('outputfile', help='the outputfile')
    parser.add_argument('-f', '--fieldnamein', 
                        help='input hdf5 fieldname <stack>')
    parser.add_argument('-g', '--fieldnameout', 
                        help='output hdf5 fieldname <stack>')
    parser.add_argument('-d', '--datatype', 
                        help='the numpy-style output datatype')
    parser.add_argument('-l', '--layout', 
                        help='the data layout for output')
    parser.add_argument('-s', '--chunksize', type=int, nargs='*', 
                        help='hdf5 chunk sizes in order of layout')
    parser.add_argument('-e', '--element_size_um', type=float, nargs='*', 
                        help='dataset element sizes in order of layout')
    parser.add_argument('-n', '--enable_duo', action='store_true', 
                        help='output the stack in nifti and hdf5')
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
    indim = len(inds.dims)
    inlayout = [d.label for d in inds.dims]
    # TODO!: handle case when no inlayout labels are present
    
    if args.layout:
        outlayout = args.layout
    elif inlayout:
        outlayout = inlayout
    else:
        outlayout = 'xyzct'[0:indim]
    
    std2in = ['xyzct'.index(l) for l in inlayout]
    std2out = ['xyzct'.index(l) for l in outlayout]
    in2out = [inlayout.index(l) for l in outlayout]
    
    if args.element_size_um:
        element_size_um = args.element_size_um
    elif all(inds.attrs['element_size_um']):
        element_size_um = [inds.attrs['element_size_um'][i] 
                           for i in in2out]
    
    if args.chunksize:
        chunksize = args.chunksize
    elif all(inds.chunks):
        chunksize = [inds.chunks[i] for i in in2out]
    
    datatype = args.datatype if args.datatype else inds.dtype
    
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
    stdsel = [[x,X],[y,Y],[z,Z],[c,C],[t,T]]
    insel = [stdsel[i] for i in std2in]
    
    b,e = path.splitext(outputfile)
    if '.nii' in outputfile:
        enable_nifti = True
        enable_hdf5 = args.enable_duo
        outhdf5 = b + '.h5'
        outnifti = outputfile
    else:
        enable_hdf5 = True
        enable_nifti = args.enable_duo
        outhdf5 = outputfile
        outnifti = b + '_' + outfield + '.nii.gz'
    
    
    
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
    if datatype != inds.dtype:
        img = img.astype(datatype,copy=False)
    if outlayout != inlayout:
        img = img.transpose(in2out)
    
    
    
    if enable_nifti:
        mat = [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
        if all(element_size_um):
            mat[0][0] = element_size_um[0]
            mat[1][1] = element_size_um[1]
            mat[2][2] = element_size_um[2]
        Nifti1Image(img, mat).to_filename(outnifti)
    
    if enable_hdf5:
        # FIXME: somehow 'a' doesn't work if file doesnt exist
        otype = 'a' if path.isfile(outputfile) else 'w'
        g = h5py.File(outhdf5, otype)
        datalayout = tuple(stdsel[i][1]-stdsel[i][0] for i in std2out)
        outds = g.create_dataset(outfield, datalayout, 
                                chunks=tuple(chunksize), dtype=datatype)
        outds[:] = img
        if all(element_size_um):
            outds.attrs['element_size_um'] = element_size_um
        for i,l in enumerate(outlayout):
            outds.dims[i].label = l
        g.close()
    
    
    

if __name__ == "__main__":
    main(sys.argv)
