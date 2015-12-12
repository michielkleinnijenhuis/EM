from os import path
import sys
from argparse import ArgumentParser

import h5py
from mpi4py import MPI
import numpy as np

def main(argv):
    """A."""
    
    parser = ArgumentParser(description=
        'Turn a supervoxel segmentation into neuron segmentation.')
    parser.add_argument('probabilities', help='')
    parser.add_argument('supervoxels', help='')
    parser.add_argument('outputpath', help='')
    parser.add_argument('-i', '--probindex', type=int, default=0, 
                        help='')
    parser.add_argument('-t', '--probthreshold', type=float, default=1, 
                        help='')
    parser.add_argument('-p', '--probvoxelcount', type=int, default=100, 
                        help='')
    parser.add_argument('-b', '--blocksize', type=int, nargs=3, 
                        default=[100,100,100], 
                        help='sizes of processing blocks')
    parser.add_argument('-m', '--usempi', action='store_true', 
                        help='use mpi4py')
    
    args = parser.parse_args()
    
    probabilities = args.probabilities
    supervoxels = args.supervoxels
    outputpath = args.outputpath
    probindex = args.probindex
    probthreshold = args.probthreshold
    probvoxelcount = args.probvoxelcount
    blocksize = args.blocksize
    usempi = args.usempi
    
    pat = '.h5'
    basename_prob, fieldname_prob = probabilities.split(pat, 1)
    basename_svox, fieldname_svox = supervoxels.split(pat, 1)
    basename_outp, fieldname_outp = outputpath.split(pat, 1)
    filepath_prob = basename_prob + pat
    filepath_svox = basename_svox + pat
    filepath_outp = basename_outp + pat
    
    x,y,z = 0,0,0
    fs = h5py.File(filepath_svox, 'r')
    X,Y,Z = fs[fieldname_svox][:,:,:].shape
    fs.close()
    
    
    nblocks = ( (X - x) / blocksize[0] * 
                (Y - y) / blocksize[1] * 
                (Z - z) / blocksize[2])
    blocks = np.array(np.mgrid[x:X:blocksize[0],
                               y:Y:blocksize[1],
                               z:Z:blocksize[2]])
    blocks = np.reshape(blocks, [3, nblocks])
    
    # FIXME: somehow 'a' doesn't work if file doesnt exist
    otype = 'a' if path.isfile(filepath_outp) else 'w'
    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # scatter the blocks
        local_nrs = scatter_series(nblocks, comm, size, rank)
        # open files
        fp = h5py.File(filepath_prob, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        fs = h5py.File(filepath_svox, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        g = h5py.File(filepath_outp, otype, driver='mpio', comm=MPI.COMM_WORLD)
        print("process", rank, "processes blocks", local_nrs)
    else:
        rank = 0
        size = 1
        local_nrs = np.array(range(0, nblocks), dtype=int)
        # open files
        fp = h5py.File(filepath_prob, 'r')
        fs = h5py.File(filepath_svox, 'r')
        g = h5py.File(filepath_outp, otype)
    
    dset = g.create_dataset(fieldname_outp, fs[fieldname_svox][:,:,:].shape, dtype='int32')
    
    for blocknr in local_nrs:
        x_local = blocks[0][blocknr]
        X_local = x_local + blocksize[0]
        y_local = blocks[1][blocknr]
        Y_local = y_local + blocksize[1]
        z_local = blocks[2][blocknr]
        Z_local = z_local + blocksize[2]
        
        p = fp[fieldname_prob][x_local:X_local,
                               y_local:Y_local,
                               z_local:Z_local,probindex] > probthreshold
        
        s = fs[fieldname_svox][x_local:X_local,
                               y_local:Y_local,
                               z_local:Z_local]
        
        forward_map = np.zeros(np.max(s) + 1, 'bool')  # FIXME? max(s) is only the local max
        x = np.bincount(s[p]) >= probvoxelcount
        forward_map[0:len(x)] = x
        segments = forward_map[s]
        
        dset[x_local:X_local,
             y_local:Y_local,
             z_local:Z_local] = segments
    
    fp.close()
    fs.close()
    g.close()

#####################
### function defs ###
#####################

def scatter_series(n, comm, size, rank):
    """Scatter a series of jobnrs over processes."""
    
    nrs = np.array(range(0, n), dtype=int)
    local_n = np.ones(size, dtype=int) * n / size
    local_n[0:n % size] += 1
    local_nrs = np.zeros(local_n[rank], dtype=int)
    displacements = tuple(sum(local_n[0:r]) for r in range(0, size))
    comm.Scatterv([nrs, tuple(local_n), displacements, 
                   MPI.SIGNED_LONG_LONG], local_nrs, root=0)
    
    return local_nrs

if __name__ == "__main__":
    main(sys.argv[1:])
