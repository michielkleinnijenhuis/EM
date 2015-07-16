#!/usr/bin/env python

import sys
import glob
import argparse
from os import path
import numpy as np
from skimage import io
from mpi4py import MPI
import h5py

def main(argv):
    
    parser = argparse.ArgumentParser(description=
        'Convert a directory of tifs to an hdf5 stack.')
    
    parser.add_argument('inputdir', help='a directory with images')
    parser.add_argument('outputfile', help='the hdf5 outputfile <*.h5>')
    parser.add_argument('-n', '--nzfills', type=int, default=4, 
                        help='the number of characters at the end that define z')
    parser.add_argument('-r', '--regex', default='*.tif', 
                        help='regular expression to select files with')
    parser.add_argument('-f', '--fieldname', default='stack', 
                        help='hdf5 fieldname <stack>')
    parser.add_argument('-d', '--datatype', 
                        help='the numpy-style output datatype')
    parser.add_argument('-o', '--ordercstyle', action='store_true', 
                        help='use c-style indexing for output, i.e. [z y x]')
    parser.add_argument('-c', '--chunksize', type=int, nargs=3, 
                        default=[20,20,20], help='hdf5 chunk sizes <x y z>')
    parser.add_argument('-e', '--element_size_um', type=float, nargs=3, 
                        default=[None,None,None], 
                        help='dataset element sizes <x y z>')
    parser.add_argument('-x', default=0, type=int, help='first x-index')
    parser.add_argument('-X', type=int, help='last x-index')
    parser.add_argument('-y', default=0, type=int, help='first y-index')
    parser.add_argument('-Y', type=int, help='last y-index')
    parser.add_argument('-z', default=0, type=int, help='first z-index')
    parser.add_argument('-Z', type=int, help='last z-index')
    
    args = parser.parse_args()
    
    inputdir = args.inputdir
    outputfile = args.outputfile
    
    nzfills = args.nzfills
    regex = args.regex
    files = glob.glob(path.join(inputdir, regex))
    (root, ext) = path.splitext(files[0])
    (head, tail) = path.split(root)
    prefix = tail[:-nzfills]
    
    firstimage = io.imread(files[0])
    x = args.x
    X = args.X if args.X else firstimage.shape[1]
    y = args.y
    Y = args.Y if args.Y else firstimage.shape[0]
    z = args.z
    Z = args.Z if args.Z else len(files)
    
    fieldname = args.fieldname
    datatype = args.datatype if args.datatype else firstimage.dtype
    ordercstyle = args.ordercstyle
    chunksize = args.chunksize
    slicechunksize = chunksize[2]
    element_size_um = args.element_size_um
    if ordercstyle:
        dimlabels = 'zyx'
        datalayout = (Z-z, Y-y, X-x)
        chunksize.reverse()
        element_size_um.reverse()
    else:
        dimlabels = 'xyz'
        datalayout = (X-x, Y-y, Z-z)
    
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    nblocks = (Z-z) / slicechunksize
    blocks = np.linspace(z, Z, nblocks, endpoint=False, dtype=int)
    local_nblocks = np.array([0])
    
    if rank == 0:
        # FIXME: somehow 'a' doesn't work if file doesnt exist
        otype = 'a' if path.isfile(outputfile) else 'w'
        f = h5py.File(outputfile, otype)
        dset = f.create_dataset(fieldname, datalayout, 
                                chunks=tuple(chunksize), dtype=datatype)
        if all(element_size_um):
            dset.attrs['element_size_um'] = element_size_um
        for i in range(0,3):
            dset.dims[i].label = dimlabels[i]
        f.close()
    
    f = h5py.File(outputfile, 'r+', driver='mpio', comm=MPI.COMM_WORLD)
    
    if rank == 0:
        local_nblocks = np.array([nblocks/size])
    comm.Bcast(local_nblocks, root=0)
    local_blocks = np.zeros(local_nblocks, dtype=int)
    comm.Scatter(blocks, local_blocks, root=0)
    
    print('process {0} will process blocks {1}'.format(rank, local_blocks))
    for startslice in local_blocks:
        print('process {0} is processing block {1}'.format(rank, startslice))
        endslice = startslice + slicechunksize
        block = []
        for imno in range(startslice, endslice):
            input_image = path.join(head, 
                                    prefix + str(imno).zfill(nzfills) + ext)
            original = io.imread(input_image).transpose([1,0])
            block.append(original[x:X,y:Y])
        blz = startslice - z
        blZ = endslice - z
        if ordercstyle:
            f[fieldname][blz:blZ,:,:] = np.array(block).transpose([0,2,1]).\
                astype(datatype,copy=False)
        else:
            f[fieldname][:,:,blz:blZ] = np.array(block).transpose([1,2,0]).\
                astype(datatype,copy=False)
    
    f.close()

if __name__ == "__main__":
    main(sys.argv)
