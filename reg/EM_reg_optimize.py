#!/usr/bin/env python

import sys
from os import path, makedirs
from argparse import ArgumentParser
import pickle
import math
import glob
from random import sample
import numpy as np

from scipy.optimize import minimize
from skimage import transform as tf

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


def main(argv):
    """Optimize transformation matrices for all pointpairs."""

    # parse arguments
    parser = ArgumentParser(description="""
        Optimize transformation matrices for all pointpairs.""")
    parser.add_argument('inputdir', help='a directory with pickles')
    parser.add_argument('outputfile', help='file to write results to')
    parser.add_argument('-r', '--regex', default='*.pickle',
                        help='regular expression to select files with')
    parser.add_argument('-f', '--fixedtile', type=int, nargs=2,
                        default=[0, 0],
                        help='fixed tile')
    parser.add_argument('-w', '--transformname', default="EuclideanTransform",
                        help='scikit-image transform class name')  # TODO: get from previous step!
    parser.add_argument('-n', '--npairs', type=int, default=100,
                        help='number of pairs to use in optimization')
    parser.add_argument('-a', '--method', default='L-BGFS-B',
                        help='minimization method')
    parser.add_argument('-i', '--maxiter', type=int, default=100,
                        help='maximum number of iterations for L-BGFS-B')
    parser.add_argument('-p', '--pc_factors', type=float, nargs=3,
                        default=[0.2 * math.pi, 0.001, 0.001],
                        help='preconditioning factors')
    parser.add_argument('-m', '--usempi', action='store_true',
                        help='use mpi4py')
    args = parser.parse_args()

    inputdir = args.inputdir
    outputfile = args.outputfile
    p = path.split(outputfile)[0]
    if not path.exists(p):
        makedirs(p)
    regex = args.regex
    fixedtile = args.fixedtile
    transformname = args.transformname
    npairs = args.npairs
    method = args.method
    maxiter = args.maxiter
    pc_factors = args.pc_factors
    usempi = args.usempi & ('mpi4py' in sys.modules)

    # load pairs
    pairs, n_slcs, n_tiles = load_pairs(inputdir, regex, npairs)

    # distribute the job
    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        SUM = MPI.SUM
        # scatter the slices
        local_nrs = scatter_series(len(pairs), comm, size, rank,
                                   MPI.SIGNED_LONG_LONG)
        # generate initilization point and send to all processes
        init_tfs = np.zeros([n_slcs, n_tiles, 3])
        if rank == 0:
            init_tfs = generate_init_tfs(pairs, n_slcs, n_tiles)
        comm.Bcast(init_tfs, root=0)
    else:
        MPI = None
        SUM = None
        comm = None
        local_nrs = np.array(range(0, len(pairs)), dtype=int)
        init_tfs = generate_init_tfs(pairs, n_slcs, n_tiles, fixedtile,
                                     transformname)

    # run minimization
    if method == 'init':
        betas = init_tfs
    else:
        init_tfs_pc = precondition_betas(init_tfs, pc_factors)
        res = minimize(obj_fun_global, init_tfs_pc,
                       args=(pairs, pc_factors, n_slcs, n_tiles, fixedtile,
                             local_nrs, usempi, comm, SUM),
                       method=method,  # L-BGFS-B  # Nelder-Mead  # Powell
                       options={'maxfun': 100000,
                                'maxiter': maxiter,
                                'disp': True})
        betas = decondition_betas(res.x, pc_factors)
        betas = np.array(betas).reshape(n_slcs, n_tiles, len(pc_factors))

    # write the optimized transformation as numpy array
    np.save(outputfile, betas)

    return betas


# ========================================================================== #
# function defs
# ========================================================================== #


def scatter_series(n, comm, size, rank, SLL):
    """Scatter a series of jobnrs over processes."""

    nrs = np.array(range(0, n), dtype=int)
    local_n = np.ones(size, dtype=int) * n / size
    local_n[0:n % size] += 1
    local_nrs = np.zeros(local_n[rank], dtype=int)
    displacements = tuple(sum(local_n[0:r]) for r in range(0, size))
    comm.Scatterv([nrs, tuple(local_n), displacements,
                   SLL], local_nrs, root=0)

    return local_nrs


def load_pairs(inputdir, regex, npairs=100):
    """Load a previously generated set of pairs."""

    pairfiles = glob.glob(path.join(inputdir, regex))

    pairs = []
    slcnr = 0
    tilenr = 0
    for pairfile in pairfiles:
        p, src, dst, model, w = pickle.load(open(pairfile, 'rb'))
        population = range(0, src.shape[0])
        try:
            pairnrs = sample(population, npairs)
        except ValueError:
            pairnrs = [i for i in population]
            print("TOO LITTLE DATA for pair in %s!" % pairfile)
        pairs.append((p, src[pairnrs, :], dst[pairnrs, :], model, w))
        slcnr = max(p[0][0], slcnr)
        tilenr = max(p[0][1], tilenr)

    return pairs, slcnr+1, tilenr+1


def generate_init_tfs(pairs, n_slcs, n_tiles, fixedtile,
                      transformname="EuclideanTransform"):
    """Find the transformation of each tile to tile[0,0]."""

    tf0 = eval("tf.%s()" % transformname)  # is this necessary?
    init_tfs = np.zeros([n_slcs, n_tiles, 3])
    for pair in pairs:
        p, _, _, model, _ = pair
        # if referenced to tile 0 & within the same slice
        if (p[0][1] == 0) & (p[0][0] == p[1][0]):
            tf1 = tf0.__add__(model)
            theta = max(tf1.params[0, 0], -1)  # is the max necessary?
            itf = [math.acos(theta), tf1.params[0, 2], tf1.params[1, 2]]
            init_tfs[p[1][0], p[1][1], :] = np.array(itf)
            # if [slcX,tile0] to [slcX-1,tile0]
        if (p[0][1] == p[1][1] == 0) & (p[1][0] - p[0][0] == 1):
            tf0 = tf0.__add__(model)

#     # recalculate wrt to fixed tile
#     if (fixedtile[0] == 0) & (fixedtile[1] == 0):
#         pass
#     else:
#         Hs = tfs_to_H(init_tfs, n_slcs, n_tiles, transformname)
#         H_fixed = Hs[fixedtile[0], fixedtile[1]]
#         Hfi = H_fixed._inv_matrix
#         H_fixed_inv = eval("tf.%s(Hfi)" % transformname)
#         for i in range(0, n_slcs):
#             for j in range(0, n_tiles):
#                 H[i, j, :, :] = H[i, j, :, :].dot(H_fixed_inv)

    return init_tfs


def tfs_to_H(tfs, n_slcs, n_tiles, transformname="EuclideanTransform"):
    """Convert a N*Mx3 array of [th,tx,ty] to NxM tf."""

    ilist = []
    for i in range(0, n_slcs):
        jlist = []
        for j in range(0, n_tiles):
            theta = tfs[i, j, 0]
            tx = tfs[i, j, 1]
            ty = tfs[i, j, 2]
            Hc = np.array([[math.cos(theta), -math.sin(theta), tx],
                           [math.sin(theta),  math.cos(theta), ty],
                           [0, 0, 1]])
            H = eval("tf.%s(Hc)" % transformname)
            jlist.append(H)
        ilist.append(jlist)

    return ilist


def precondition_betas(betas, pc_factors):
    """Precondition the parameters before minimization."""

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


def obj_fun_global(pars, pairs, pc_factors, n_slcs, n_tiles, fixedtile,
                   local_nrs, usempi=0, comm=None, SUM=None):
    """Calculate sum of squared error distances of keypoints."""

    # construct the transformation matrix for each slc/tile.
    H = np.empty([n_slcs, n_tiles, 3, 3])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            imno = i * n_tiles + j
            # TODO: option to choose other reference tile (e.g. m000_0250.tif!)
            if (i == fixedtile[0]) & (j == fixedtile[1]):
                H[i, j, :, :] = np.identity(3)
            else:
                th = pars[(imno+1)*3-3] / pc_factors[0]
                tx = pars[(imno+1)*3-2] / pc_factors[1]
                ty = pars[(imno+1)*3-1] / pc_factors[2]
                H[i, j, :, :] = np.array([[math.cos(th), -math.sin(th), tx],
                                          [math.sin(th),  math.cos(th), ty],
                                          [0, 0, 1]])

    wses = np.empty([0, 2])
    for i in local_nrs:
        p, s, d, _, w = pairs[i]
        # homogenize pointsets
        d = np.c_[d, np.ones(d.shape[0])]
        s = np.c_[s, np.ones(s.shape[0])]
        # transform d/s points to image000 space
        d = d.dot(H[p[0][0], p[0][1], :, :].T)[:, :2]
        s = s.dot(H[p[1][0], p[1][1], :, :].T)[:, :2]
        wse = w * (d - s)**2
        # concatenate the weighted squared errors of all the pairs
        wses = np.r_[wses, wse]
    # and sum
    sse = np.sum(wses)

    if usempi:
        sse = comm.allreduce(sse, op=SUM)

    return sse

if __name__ == "__main__":
    main(sys.argv)
