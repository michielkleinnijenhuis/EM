

import os
import pickle

outputdir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_regpointpairs/reg_d4"

pairfile = os.path.join(outputdir, "pair_c1_d4_s0000-t0_s0000-t1.pickle")
with open(pairfile, 'rb') as f:
    p, src, dst, model, w = pickle.load(f)



find /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_regpointpairs/reg -type f -newermt "Aug 30 13:00" -delete


import os
import numpy as np
import math

outputdir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU_regpointpairs/reg_d4"
betasfile = os.path.join(outputdir, "betas_o1_d4.npy")
betas = np.load(betasfile)

n_slcs = betas.shape[0]
n_tiles = betas.shape[1]

def tfs_to_H(tfs, n_slcs, n_tiles):
    """Convert a N*Mx3 array of [th,tx,ty] to NxM 3x3 tf matrices."""
    H = np.empty([n_slcs, n_tiles, 3, 3])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            theta = tfs[i,j,0]
            tx = tfs[i,j,1]
            ty = tfs[i,j,2]
            H[i,j,:,:] = np.array([[math.cos(theta), -math.sin(theta), tx],
                                   [math.sin(theta),  math.cos(theta), ty],
                                   [0,0,1]])
    return H

H = tfs_to_H(betas, n_slcs, n_tiles)

Hnew = np.dot(H1, S)
