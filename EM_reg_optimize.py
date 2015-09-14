#!/usr/bin/env python

import sys
from os import path, makedirs
from argparse import ArgumentParser
import pickle
import math

import numpy as np

from skimage import io
from matplotlib.pylab import c_, r_
from scipy.optimize import minimize
from skimage.transform import SimilarityTransform

from mpi4py import MPI
# from time import time

def main(argv):
    """."""
    
    # parse arguments
    parser = ArgumentParser(description=
        'Generate matching point-pairs for stack registration.')
    parser.add_argument('datadir', help='a directory with images')
    parser.add_argument('-o', '--outputdir', 
                        help='directory to write results')
    parser.add_argument('-u', '--upairsfile', 
                        help='pickle with the unique pairs')
    parser.add_argument('-t', '--n_tiles', type=int, default=4, 
                        help='the number of tiles in the montage')
    parser.add_argument('-c', '--offsets', type=int, default=2, 
                        help='the number of sections in z to consider')
    parser.add_argument('-d', '--downsample_factor', type=int, default=1, 
                        help='the factor to downsample the images by')
    parser.add_argument('-i', '--maxiter', type=int, default=100, 
                        help='maximum number of iterations for L-BGFS-B')
    parser.add_argument('-p', '--pc_factors', type=float, nargs=3, 
                        default=[0.2*math.pi, 0.001, 0.001], 
                        help='the number of initial keypoints to generate')
    parser.add_argument('-m', '--usempi', action='store_true', 
                        help='use mpi4py')
    args = parser.parse_args()
    
    datadir = args.datadir
    if not args.outputdir:
        outputdir = datadir
    else:
        outputdir = args.outputdir
        if not path.exists(outputdir):
            makedirs(outputdir)
    upairsfile = args.upairsfile
    n_tiles = args.n_tiles
    offsets = args.offsets
    downsample_factor = args.downsample_factor
    maxiter = args.maxiter
    pc_factors = args.pc_factors
    usempi = args.usempi
    
    if not upairsfile:
        # get the image collection
        imgs = io.ImageCollection(path.join(datadir,'*.tif'))
        n_slcs = len(imgs) / n_tiles
        imgs = [imgs[(slc+1)*n_tiles-n_tiles:slc*n_tiles+4] 
                for slc in range(0, n_slcs)]
        
        # determine which pairs of images to process
        # NOTE: ['d2', 1, 2] non-overlapping for M3 dataset
        connectivities = [['z', 0, 0],['z', 1, 1],['z', 2, 2],['z', 3, 3],
                          ['y', 0, 2],['y', 1, 3],
                          ['x', 0, 1],['x', 2, 3],
                          ['d1', 0, 3]]
        unique_pairs = generate_unique_pairs(n_slcs, offsets, connectivities)
    else:
        unique_pairs = pickle.load(open(upairsfile, 'rb'))
        n_slcs = unique_pairs[-1][0][0] + 1
    
    # load, initialize, precondition, minimize, decondition, save betas
    pairs = load_pairs(outputdir, offsets, downsample_factor, unique_pairs)
    
    
    init_tfs = np.zeros([n_slcs, n_tiles, 3])
    
    n = len(pairs)
    nrs = np.array(range(0, n), dtype=int)
    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # scatter the slices
        local_n = np.ones(size, dtype=int) * n / size
        local_n[0:n % size] += 1
        local_nrs = np.zeros(local_n[rank], dtype=int)
        displacements = tuple(sum(local_n[0:r]) for r in range(0, size))
        comm.Scatterv([nrs, tuple(local_n), displacements, 
                       MPI.SIGNED_LONG_LONG], local_nrs, root=0)
        
        if rank == 0:
            init_tfs = generate_init_tfs(pairs, n_slcs, n_tiles)
        comm.Bcast(init_tfs, root=0)
    else:
        comm = None
        
        local_nrs = nrs
        
        init_tfs = generate_init_tfs(pairs, n_slcs, n_tiles)
    
    init_tfs_pc = precondition_betas(init_tfs, pc_factors)
    
    res = minimize(obj_fun_global, init_tfs_pc, 
                   args=(pairs, pc_factors, n_slcs, n_tiles, 
                         local_nrs, usempi, comm), 
                   method='L-BFGS-B',   # Nelder-Mead  # Powell
                   options={'maxfun':100000, 'maxiter':maxiter, 'disp':True})
    
    betas = decondition_betas(res.x, pc_factors)
    betas = np.array(betas).reshape(n_slcs, n_tiles, len(pc_factors))
    betasfile = path.join(outputdir, 'betas' + 
                          '_o' + str(offsets) + 
                          '_s' + str(downsample_factor) + '.npy')
    np.save(betasfile, betas)
    
    return 0


#####################
### function defs ###
#####################

def generate_unique_pairs(n_slcs, offsets, connectivities):
    """Get a list of unique pairs with certain connectivity."""
    # list of [[slcIm1, tileIm1], [slcIm2, tileIm2], 'type']
    all_pairs = [[[slc,c[1]], [slc+o,c[2]], c[0]] 
                 for slc in range(0, n_slcs) 
                 for o in range(0, offsets+1) 
                 for c in connectivities]
    unique_pairs = []
    for pair in all_pairs:
        if (([pair[1],pair[0],pair[2]] not in unique_pairs) & 
            (pair[0] != pair[1]) & 
            (pair[1][0] != n_slcs)):
            unique_pairs.append(pair)
    
    return unique_pairs


def load_pairs(outputdir, offsets, downsample_factor, unique_pairs):
    """Load a previously generated set of pairs."""
    pairs = []
    for p in unique_pairs:
        pairstring = 'pair' + \
                     '_c' + str(offsets) + \
                     '_d' + str(downsample_factor) + \
                     '_s' + str(p[0][0]).zfill(4) + \
                     '-t' + str(p[0][1]) + \
                     '_s' + str(p[1][0]).zfill(4) + \
                     '-t' + str(p[1][1])
        pairfile = path.join(outputdir, pairstring + '.pickle')
        p, src, dst, model, w = pickle.load(open(pairfile, 'rb'))
        pairs.append((p, src, dst, model, w))
    
    return pairs

def generate_init_tfs(pairs, n_slcs, n_tiles):
    """Find the transformation of each tile to tile[0,0]."""
    tf0 = SimilarityTransform()
    init_tfs = np.zeros([n_slcs, n_tiles, 3])
    for pair in pairs:
        p, _, _, model, _ = pair
        if (p[0][1] == 0) & (p[0][0] == p[1][0]):  # if referenced to tile 0 & within the same slice
            tf1 = tf0.__add__(model)
            if tf1.params[0,0] > 0:  # FIXME!!! with RigidTransform
                theta = min(tf1.params[0,0], 1)
            else:
                theta = max(tf1.params[0,0], -1)
            itf = [math.acos(theta), tf1.params[0,2], tf1.params[1,2]]
            init_tfs[p[1][0],p[1][1],:] = np.array(itf)
        if (p[0][1] == p[1][1] == 0) & (p[1][0] - p[0][0] == 1):  # if [slcX,tile0] to [slcX-1,tile0]
            tf0 = tf0.__add__(model)
    
    return init_tfs

def precondition_betas(betas, pc_factors):
    betas = betas.reshape(betas.shape[0] * betas.shape[1], len(pc_factors))
    betas_tmp = [list(b * pc_factors) for b in betas]
    betas_pc = [item for sublist in betas_tmp for item in sublist]
    
    return betas_pc

def decondition_betas(betas_pc, pc_factors):
    """Decondition the parameters after minimization."""
    l = len(pc_factors)
    betas_pc = np.array(betas_pc).reshape(betas_pc.size/l, l)
    betas_tmp = [list(b / pc_factors) for b in betas_pc]
    betas = [item for sublist in betas_tmp for item in sublist]
    
    return betas

def obj_fun_global(pars, pairs, pc_factors, n_slcs, n_tiles, local_nrs, usempi=0, comm=None):
    """Calculate sum of squared error distances of keypoints."""
    # construct the transformation matrix for each slc/tile.
    H = np.empty([n_slcs, n_tiles, 3, 3])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            imno = i * n_tiles + j
            if i+j == 0:
                H[i,j,:,:] = np.identity(3)
            else:
                theta = pars[(imno+1)*3-3] / pc_factors[0]
                tx = pars[(imno+1)*3-2] / pc_factors[1]
                ty = pars[(imno+1)*3-1] / pc_factors[2]
                H[i,j,:,:] = np.array([[math.cos(theta), -math.sin(theta), tx],
                                       [math.sin(theta),  math.cos(theta), ty],
                                       [0,0,1]])
    
#     pair_ssestart = time()
    wses = np.empty([0,2])
    local_pairs = [(i,up) for i,up in enumerate(pairs) 
                   if i in local_nrs]
    for i, (p,s,d,_,w) in local_pairs:
        # homogenize pointsets
        d = c_[d, np.ones(d.shape[0])]
        s = c_[s, np.ones(s.shape[0])]
        # transform d/s points to image000 space
        d = d.dot(H[p[0][0],p[0][1],:,:].T)[:,:2]
        s = s.dot(H[p[1][0],p[1][1],:,:].T)[:,:2]
        wse = w * (d - s)**2
        # concatenate the weighted squared errors of all the pairs
        wses = r_[wses, wse]
    # and sum
    sse = np.sum(wses)
    
    if usempi:
        sse = comm.allreduce(sse, op = MPI.SUM)
    
#     print('sse calculated in: %6.2f s' % (time() - pair_ssestart))
    return sse

if __name__ == "__main__":
    main(sys.argv)
