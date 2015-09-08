#!/usr/bin/env python

import sys
from os import path, makedirs
from argparse import ArgumentParser
import pickle
import math

import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import gaussian
from mpi4py import MPI

from skimage.transform import rescale
from skimage import io
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.feature import plot_matches
from skimage.transform import SimilarityTransform


def main(argv):
    """."""
    
    # parse arguments
    parser = ArgumentParser(description=
        'Generate matching point-pairs for stack registration.')
    parser.add_argument('datadir', help='a directory with images')
    parser.add_argument('-o', '--outputdir', 
                        help='directory to write results')
    parser.add_argument('-t', '--n_tiles', type=int, default=4, 
                        help='the number of tiles in the montage')
    parser.add_argument('-c', '--offsets', type=int, default=2, 
                        help='the number of sections in z to consider')
    parser.add_argument('-s', '--subsample_factor', type=int, default=1, 
                        help='the factor to downsample the images by')
    parser.add_argument('-f', '--overlap_fraction', type=float, nargs=2, 
                        default=[0.1,0.1], help='section overlap in [y,x]')
    parser.add_argument('-k', '--n_keypoints', type=int, default=10000, 
                        help='the number of initial keypoints to generate')
    parser.add_argument('-p', '--plotpairs', action='store_true', 
                        help='create plots of point-pairs')
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
    n_tiles = args.n_tiles
    offsets = args.offsets
    subsample_factor = args.subsample_factor
    overlap_fraction = args.overlap_fraction
    n_keypoints = args.n_keypoints
    usempi = args.usempi
    plotpairs = args.plotpairs
    
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
    
    
    # get the feature class
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=0.05)
    # get the weighing kernel in z
    k = gaussian(offsets*2+1, 1, sym=True)
    
    npairs = len(unique_pairs)
    pairnrs = np.array(range(0, npairs), dtype=int)
    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # scatter the pairs
        local_npairs = np.ones(size, dtype=int) * npairs / size
        local_npairs[0:npairs % size] += 1
        local_pairnrs = np.zeros(local_npairs[rank], dtype=int)
        displacements = tuple(sum(local_npairs[0:r]) for r in range(0, size))
        comm.Scatterv([pairnrs, tuple(local_npairs), displacements, 
                       MPI.SIGNED_LONG_LONG], local_pairnrs, root=0)
    else:
        local_pairnrs = pairnrs
    
    # process the assigned pairs
    local_pairs = [(i,up) for i,up in enumerate(unique_pairs) 
                   if i in local_pairnrs]
    for pid, p in local_pairs:
        # FIXME: handle case where get_pair fails (no inliers)
        pair = get_pair(datadir, imgs, p, pid, offsets, 
                        subsample_factor, overlap_fraction, orb, k, plotpairs)
        pairfile = path.join(datadir, 'pairs' + 
                             '_o' + str(offsets) + 
                             '_s' + str(subsample_factor) + 
                             '_p' + str(pid).zfill(4) + '.pickle')
        pickle.dump(pair, open(pairfile, 'wb'))
    
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

def subsample_images(p, imgs, subsample_factor):
    """Subsample images with subsample_factor"""
    if subsample_factor > 1:
        full_im1 = rescale(imgs[p[0][0]][p[0][1]], 1./subsample_factor)
        full_im2 = rescale(imgs[p[1][0]][p[1][1]], 1./subsample_factor)
    else:
        full_im1 = imgs[p[0][0]][p[0][1]]
        full_im2 = imgs[p[1][0]][p[1][1]]
    return full_im1, full_im2

def select_imregions(ptype, full_im1, full_im2, overlap_pixels):
    """Select image regions to extract keypoints from."""
    if ptype == 'z':
        im1 = full_im1
        im2 = full_im2
    elif ptype in 'y':
        y1 = full_im1.shape[0] - overlap_pixels[0]
        y2 = overlap_pixels[0]
        im1 = full_im1[y1:,:]
        im2 = full_im2[:y2,:]
    elif ptype in 'x':
        x1 = full_im1.shape[1] - overlap_pixels[1]
        x2 = overlap_pixels[1]
        im1 = full_im1[:,x1:]
        im2 = full_im2[:,:x2]
    elif ptype in 'd1':
        x1 = full_im1.shape[1] - 2 * overlap_pixels[1]
        y1 = full_im1.shape[0] - 2 * overlap_pixels[0]
        x2 = 2 * overlap_pixels[1]
        y2 = 2 * overlap_pixels[0]
        im1 = full_im1[y1:,x1:]
        im2 = full_im2[:y2,:x2]
    elif ptype in 'd2':
        x1 = full_im1.shape[1] - 2 * overlap_pixels[1]
        y1 = 2 * overlap_pixels[0]
        x2 = 2 * overlap_pixels[1]
        y2 = full_im2.shape[0] - 2 * overlap_pixels[0]
        im1 = full_im1[:y1,x1:]
        im2 = full_im2[y2:,:x2]
    return im1, im2

def get_keypoints(orb, im):
    """Get matching keypoints."""
    orb.detect_and_extract(im)
    kp = orb.keypoints
    ds = orb.descriptors
    return kp, ds

def reset_imregions(ptype, kp_im1, kp_im2, overlap_pixels, imshape):
    """Transform keypoints back to full image space."""
    if ptype in 'z':
        pass
    elif ptype in 'y':
        kp_im1[:,0] += imshape[0] - overlap_pixels[0]
    elif ptype in 'x':
        kp_im1[:,1] += imshape[1] - overlap_pixels[1]
    elif ptype in 'd1':
        kp_im1[:,0] += imshape[0] - 2 * overlap_pixels[0]
        kp_im1[:,1] += imshape[1] - 2 * overlap_pixels[1]
    elif ptype in 'd2':
        kp_im1[:,0] += imshape[0] - 2 * overlap_pixels[0]
        kp_im2[:,1] += imshape[1] - 2 * overlap_pixels[1]
    return kp_im1, kp_im2

def plot_pair_ransac(datadir, p, full_im1, full_im2, kp_im1, kp_im2, matches, inliers):
    """Create plots of orb keypoints vs. ransac inliers."""
    fig, (ax1,ax2) = plt.subplots(2,1)
    plot_matches(ax1, full_im1, full_im2, kp_im1, kp_im2, 
                 matches, only_matches=True)
    ax1.axis('off')
    plot_matches(ax2, full_im1, full_im2, kp_im1, kp_im2, 
                 matches[inliers], only_matches=True)
    ax2.axis('off')
    plotdir = path.join(datadir, 'plotpairs')
    if not path.exists(plotdir):
        makedirs(plotdir)
    fig.savefig(path.join(plotdir, 'pair_s' + 
                          str(p[0][0]).zfill(4) + '-t' + 
                          str(p[0][1]) + '_s' +
                          str(p[1][0]).zfill(4) + '-t' + 
                          str(p[1][1]) + '.tif'))
    plt.close(fig)

def get_pair(datadir, imgs, p, pid, offsets, subsample_factor, overlap_fraction, orb, k, plotpairs):
    """Create inlier keypoint pairs."""
    
    pair_tstart = time()
    
    overlap_pixels = [int(math.ceil(d * overlap_fraction[i] * 1/subsample_factor)) 
                      for i,d in enumerate(imgs[0][0].shape)]
    
    f1, f2 = subsample_images(p, imgs, subsample_factor)
    p1, p2 = select_imregions(p[2], f1, f2, overlap_pixels)
    kp1, de1 = get_keypoints(orb, p1)
    kp2, de2 = get_keypoints(orb, p2)
    kp1, kp2 = reset_imregions(p[2], kp1, kp2, overlap_pixels, f1.shape)
    
    matches = match_descriptors(de1, de2, cross_check=True)
    dst = kp1[matches[:, 0]][:, ::-1]
    src = kp2[matches[:, 1]][:, ::-1]
    model, inliers = ransac((src, dst), SimilarityTransform, min_samples=4, 
                            residual_threshold=2, max_trials=300)
    # FIXME: is there no rigid model in scikit-image??? # this is needed for input to RANSAC
    
    w = k[offsets - (p[1][0] - p[0][0])]
    
    if plotpairs:
        plot_pair_ransac(datadir, p, f1, f2, kp1, kp2, matches, inliers)
    
    print('Pair %04d done in: %.2f s; matches: %05d; inliers: %05d' 
          % (pid, time() - pair_tstart, matches, inliers))
    return (p, src[inliers], dst[inliers], model, w)

if __name__ == "__main__":
    main(sys.argv)
