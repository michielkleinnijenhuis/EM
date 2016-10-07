#!/usr/bin/env python

import sys
from os import path, makedirs
from argparse import ArgumentParser
import pickle
import math

import numpy as np
from time import time
from scipy.signal import gaussian

from skimage import io
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.measure import ransac
from skimage import transform as tf

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


def main(argv):
    """Generate matching point-pairs for stack registration."""

    # parse arguments
    parser = ArgumentParser(description="""
        Generate matching point-pairs for stack registration.""")
    parser.add_argument('imgdir', help='a directory with images')
    parser.add_argument('outputdir', help='directory to write results')
    parser.add_argument('-u', '--pairs',
                        help='pickle with pairs to process')
    parser.add_argument('-c', '--connectivityfile',
                        help='file containing connectivity specification')
    parser.add_argument('-t', '--n_tiles', type=int, default=4,
                        help='the number of tiles in the montage')
    parser.add_argument('-f', '--overlap_fraction', type=float, nargs=2,
                        default=[0.1, 0.1], help='section overlap in [y,x]')
    parser.add_argument('-o', '--offsets', type=int, default=1,
                        help='the number of sections in z to consider')
    parser.add_argument('-d', '--downsample_factor', type=int, default=1,
                        help='the factor to downsample the images by')
    parser.add_argument('-w', '--transformname', default="EuclideanTransform",
                        help='scikit-image transform class name')
    parser.add_argument('-k', '--n_keypoints', type=int, default=10000,
                        help='the number of initial keypoints to generate')
    parser.add_argument('-r', '--residual_threshold', type=float,
                        default=2, help='inlier threshold for ransac')
    parser.add_argument('-n', '--num_inliers', type=int, default=None,
                        help='the number of ransac inliers to look for')
    parser.add_argument('-p', '--plotpairs', action='store_true',
                        help='create plots of point-pairs')
    parser.add_argument('-m', '--usempi', action='store_true',
                        help='use mpi4py')
    args = parser.parse_args()

    imgdir = args.imgdir
    outputdir = args.outputdir
    if not path.exists(outputdir):
        makedirs(outputdir)
    confilename = args.connectivityfile
    n_tiles = args.n_tiles
    overlap_fraction = args.overlap_fraction
    offsets = args.offsets
    ds = args.downsample_factor
    transformname = args.transformname
    n_keypoints = args.n_keypoints
    residual_threshold = args.residual_threshold
    num_inliers = args.num_inliers
    plotpairs = args.plotpairs
    usempi = args.usempi & ('mpi4py' in sys.modules)

    # get the image collection (reshaped to n_slcs x n_tiles)
    imgs = io.ImageCollection(path.join(imgdir, '*.tif'))
    n_slcs = len(imgs) / n_tiles
    imgs = [imgs[(slc + 1) * n_tiles - n_tiles:slc * n_tiles + n_tiles]
            for slc in range(0, n_slcs)]

    # determine which pairs of images to process
    connectivities = read_connectivities(confilename)
    unique_pairs = generate_unique_pairs(n_slcs, offsets, connectivities)
    upairstring = 'unique_pairs' + '_c' + str(offsets) + '_d' + str(ds)
    pairfile = path.join(outputdir, upairstring + '.pickle')
    with open(pairfile, 'wb') as f:
        pickle.dump(unique_pairs, f)
    if args.pairs:
        try:
            with open(args.pairs, 'rb') as f:
                pairs = pickle.load(f)
        except:
            pairs = find_missing_pairs(outputdir, unique_pairs, offsets, ds)
    else:
        pairs = unique_pairs

    # get the feature class
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=0.08,
              n_scales=8, downscale=1.2)

    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # scatter the pairs
        local_nrs = scatter_series(len(pairs), comm, size, rank,
                                   MPI.SIGNED_LONG_LONG)
    else:
        local_nrs = np.array(range(0, len(pairs)), dtype=int)

    # process the assigned pairs
    allpairs = []
    for i in local_nrs:
        pair = get_pair(outputdir, imgs, pairs[i], offsets,
                        ds, overlap_fraction, orb, plotpairs,
                        residual_threshold, num_inliers, transformname)
        # FIXME: handle case where get_pair fails
        allpairs.append(pair)

    return allpairs


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


def read_connectivities(confilename):
    """Read pair connectivities from file.

    specified for each pair per line as:
    type imno1 imno2
    where type is one of x y tlbr trbl
    connectivities = [['z', 0, 0], ['z', 1, 1], ['z', 2, 2], ['z', 3, 3],
                      ['y', 0, 2], ['y', 1, 3],
                      ['x', 0, 1], ['x', 2, 3],
                      ['tlbr', 0, 3], ['trbl', 1, 2]]
    # NOTE: ['trbl', 1, 2] non-overlapping for M3 dataset
    """

    with open(confilename) as f:
        con = [line.rstrip('\n').split() for line in f]

    con = [[c[0], int(c[1]), int(c[2])] for c in con]

    return con


def generate_pairstring(offsets, ds, p):
    """Get the pair identifier."""

    pairstring = 'pair' + \
                 '_c' + str(offsets) + \
                 '_d' + str(ds) + \
                 '_s' + str(p[0][0]).zfill(4) + \
                 '-t' + str(p[0][1]) + \
                 '_s' + str(p[1][0]).zfill(4) + \
                 '-t' + str(p[1][1])

    return pairstring


def generate_unique_pairs(n_slcs, offsets, connectivities):
    """Get a list of unique pairs with certain connectivity.

    list is of the form [[slcIm1, tileIm1], [slcIm2, tileIm2], 'type']
    """

    all_pairs = [[[slc, c[1]], [slc+o, c[2]], c[0]]
                 for slc in range(0, n_slcs)
                 for o in range(0, offsets+1)
                 for c in connectivities]
    unique_pairs = []
    for pair in all_pairs:
        if (([pair[1], pair[0], pair[2]] not in unique_pairs) &
                (pair[0] != pair[1]) &
                (pair[1][0] != n_slcs)):
            unique_pairs.append(pair)

    return unique_pairs


def find_missing_pairs(directory, unique_pairs, offsets, ds):
    """Get a list of missing pairs.

    list is of the form [[slcIm1, tileIm1], [slcIm2, tileIm2], 'type']
    """

    missing_pairs = []
    for p in unique_pairs:
        pairstring = generate_pairstring(offsets, ds, p)
        try:
            open(path.join(directory, pairstring + ".pickle"), 'rb')
        except:
            missing_pairs.append(p)

    return missing_pairs


def downsample_images(p, imgs, ds):
    """Subsample images with downsample_factor"""

    if ds > 1:
        full_im1 = tf.rescale(imgs[p[0][0]][p[0][1]], 1./ds)
        full_im2 = tf.rescale(imgs[p[1][0]][p[1][1]], 1./ds)
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
        im1 = full_im1[y1:, :]
        im2 = full_im2[:y2, :]
    elif ptype in 'x':
        x1 = full_im1.shape[1] - overlap_pixels[1]
        x2 = overlap_pixels[1]
        im1 = full_im1[:, x1:]
        im2 = full_im2[:, :x2]
    elif ptype in 'tlbr':  # TopLeft - BottomRight
        x1 = full_im1.shape[1] - 2 * overlap_pixels[1]
        y1 = full_im1.shape[0] - 2 * overlap_pixels[0]
        x2 = 2 * overlap_pixels[1]
        y2 = 2 * overlap_pixels[0]
        im1 = full_im1[y1:, x1:]
        im2 = full_im2[:y2, :x2]
    elif ptype in 'trbl':  # TopRight - BottomLeft
        x1 = full_im1.shape[1] - 2 * overlap_pixels[1]
        y1 = 2 * overlap_pixels[0]
        x2 = 2 * overlap_pixels[1]
        y2 = full_im2.shape[0] - 2 * overlap_pixels[0]
        im1 = full_im1[:y1, x1:]
        im2 = full_im2[y2:, :x2]

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
        kp_im1[:, 0] += imshape[0] - overlap_pixels[0]
    elif ptype in 'x':
        kp_im1[:, 1] += imshape[1] - overlap_pixels[1]
    elif ptype in 'tlbr':  # TopLeft - BottomRight
        kp_im1[:, 0] += imshape[0] - 2 * overlap_pixels[0]
        kp_im1[:, 1] += imshape[1] - 2 * overlap_pixels[1]
    elif ptype in 'trbl':  # TopRight - BottomLeft
        kp_im1[:, 0] += imshape[0] - 2 * overlap_pixels[0]
        kp_im2[:, 1] += imshape[1] - 2 * overlap_pixels[1]
    return kp_im1, kp_im2


def plot_pair_ransac(outputdir, pairstring, p, full_im1, full_im2,
                     kp_im1, kp_im2, matches, inliers):
    """Create plots of orb keypoints vs. ransac inliers."""

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_matches(ax1, full_im1, full_im2, kp_im1, kp_im2,
                 matches, only_matches=True)
    ax1.axis('off')
    plot_matches(ax2, full_im1, full_im2, kp_im1, kp_im2,
                 matches[inliers], only_matches=True)
    ax2.axis('off')
    plotdir = path.join(outputdir, 'plotpairs')
    if not path.exists(plotdir):
        makedirs(plotdir)
    fig.savefig(path.join(plotdir, pairstring))
    plt.close(fig)


def get_pair(outputdir, imgs, p, offsets, ds,
             overlap_fraction, orb,
             plotpairs=0, res_th=10, num_inliers=100,
             transformname="EuclideanTransform"):
    """Create inlier keypoint pairs."""

    pair_tstart = time()

    overlap_pixels = [int(math.ceil(d * of * 1/ds))
                      for d, of in zip(imgs[0][0].shape, overlap_fraction)]

    f1, f2 = downsample_images(p, imgs, ds)
    p1, p2 = select_imregions(p[2], f1, f2, overlap_pixels)
    kp1, de1 = get_keypoints(orb, p1)
    kp2, de2 = get_keypoints(orb, p2)
    kp1, kp2 = reset_imregions(p[2], kp1, kp2, overlap_pixels, f1.shape)

    matches = match_descriptors(de1, de2, cross_check=True)
    dst = kp1[matches[:, 0]][:, ::-1]
    src = kp2[matches[:, 1]][:, ::-1]
    transform = eval("tf.%s" % transformname)
    model, inliers = ransac((src, dst), transform, min_samples=4,
                            residual_threshold=res_th,
                            max_trials=1000, stop_sample_num=num_inliers)

    # get the weighing kernel in z
    k = gaussian(offsets*2+1, 1, sym=True)
    w = k[offsets - (p[1][0] - p[0][0])]

    # transform from downsampled space to full
    S = np.array([[ds, 0, 0],
                  [0, ds, 0],
                  [0, 0, 1]])
    s = np.c_[src, np.ones(src.shape[0])].dot(S)[inliers, :2]
    d = np.c_[dst, np.ones(dst.shape[0])].dot(S)[inliers, :2]
    pair = (p, s, d, model, w)

    pairstring = generate_pairstring(offsets, ds, p)
    pairfile = path.join(outputdir, pairstring + '.pickle')
    pickle.dump(pair, open(pairfile, 'wb'))
    if plotpairs:
        plot_pair_ransac(outputdir, pairstring, p,
                         f1, f2, kp1, kp2, matches, inliers)

    print('%s done in: %6.2f s; matches: %05d; inliers: %05d'
          % (pairstring, time() - pair_tstart, len(matches), np.sum(inliers)))

    return pair

if __name__ == "__main__":
    main(sys.argv)
