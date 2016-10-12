#!/usr/bin/env python

import sys
from os import path, makedirs
from argparse import ArgumentParser

import numpy as np

from skimage import io

from skimage.transform import rescale
import math
from scipy.ndimage.interpolation import map_coordinates
from scipy.misc import imsave
from skimage import transform as tf

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


def main(argv):
    """Transform, blend and write images."""

    # parse arguments
    parser = ArgumentParser(description="""
        Transform, blend and write images.""")
    parser.add_argument('inputdir',
                        help='a directory with original images')
    parser.add_argument('betasfile',
                        help='files with transformations [theta,tx,ty]: \
                        numpy array n_slcs x n_tiles x 3')
    parser.add_argument('outputdir',
                        help='directory to write results')
    parser.add_argument('-d', '--downsample_factor', type=int, default=1,
                        help='the factor to downsample the images by')
    parser.add_argument('-i', '--interpolation_order', type=int, default=1,
                        help='the order for the interpolation')
    parser.add_argument('-f', '--fixedtile', type=int, nargs=2, default=[0, 0],
                        help='fixed tile')
    parser.add_argument('-w', '--transformname', default="EuclideanTransform",
                        help='scikit-image transform class name')
    parser.add_argument('-m', '--usempi', action='store_true',
                        help='use mpi4py')
    args = parser.parse_args()

    inputdir = args.inputdir
    betasfile = args.betasfile
    outputdir = args.outputdir
    if not path.exists(outputdir):
        makedirs(outputdir)
    downsample_factor = args.downsample_factor
    interpolation_order = args.interpolation_order
    fixedtile = args.fixedtile
    transformname = args.transformname
    usempi = args.usempi & ('mpi4py' in sys.modules)

    # load betas and make transformation matrices
    betas = np.load(betasfile)
    n_slcs = betas.shape[0]
    n_tiles = betas.shape[1]
    H = tfs_to_H(betas, n_slcs, n_tiles)
    H = recalculate_transforms(H, fixedtile, transformname, n_slcs, n_tiles)

    # get the image collection
    imgs = io.ImageCollection(path.join(inputdir, '*.tif'))
    imgs = [imgs[(slc+1)*n_tiles-n_tiles:slc*n_tiles+n_tiles]
            for slc in range(0, n_slcs)]

    # create an distance image for blending weights
    imshape = [s/downsample_factor for s in imgs[0][0].shape]
    distance_image = create_distance_image(imshape)

    # create the canvas coordinate list
    canvas_bounds = get_canvas_bounds(imshape, n_slcs, n_tiles, H)
    # [[xmin,ymin],[xmax,ymax]]
    x = np.linspace(canvas_bounds[0, 0], canvas_bounds[1, 0],
                    canvas_bounds[1, 0] - canvas_bounds[0, 0])
    y = np.linspace(canvas_bounds[0, 1], canvas_bounds[1, 1],
                    canvas_bounds[1, 1] - canvas_bounds[0, 1])
    X, Y = np.meshgrid(x, y)  # [y,x] shape
    coordinates = np.c_[np.ndarray.flatten(X), np.ndarray.flatten(Y),
                        np.ones_like(np.ndarray.flatten(X))]  # [x,y]
    canvasshape = X.shape

    # distribute the job
    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # scatter the slices
        local_nrs = scatter_series(n_slcs, comm, size, rank)
    else:
        local_nrs = np.array(range(0, n_slcs), dtype=int)

    # process the assigned pairs
    local_slcs = [(i, im) for i, im in enumerate(imgs) if i in local_nrs]
    blend_tiles(outputdir, n_tiles, local_slcs, canvasshape,
                coordinates, H, distance_image,
                downsample_factor, interpolation_order)


# ========================================================================== #
# function defs
# ========================================================================== #


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


def create_distance_image(imshape):
    """Create an image coding for every pixel the distance from the edge."""

    # TODO: get a Gaussian weighting on the image???
    xd = np.linspace(-imshape[1]/2, imshape[1]/2, imshape[1])
    yd = np.linspace(-imshape[0]/2, imshape[0]/2, imshape[0])
    Xd, Yd = np.meshgrid(xd, yd, sparse=True)
    distance_image = np.minimum(-(np.abs(Xd) - imshape[1]/2),
                                -(np.abs(Yd) - imshape[0]/2))

    return distance_image


def tfs_to_H(tfs, n_slcs, n_tiles):
    """Convert a N*Mx3 array of [th,tx,ty] to NxM 3x3 tf matrices."""

    H = np.empty([n_slcs, n_tiles, 3, 3])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            theta = tfs[i, j, 0]
            tx = tfs[i, j, 1]
            ty = tfs[i, j, 2]
            H[i, j, :, :] = np.array([[math.cos(theta), -math.sin(theta), tx],
                                      [math.sin(theta),  math.cos(theta), ty],
                                      [0, 0, 1]])

    return H


def recalculate_transforms(H00, fixedtile, transformname, n_slcs, n_tiles):
    """Recompute transformation matrices wrt a different reference tile."""

    H = np.empty([n_slcs, n_tiles, 3, 3])
    if (fixedtile[0] == 0) & (fixedtile[1] == 0):
        H = H00
    else:
        Hf = H00[fixedtile[0], fixedtile[1]]
        Hf = eval("tf.%s(Hf)" % transformname)
        Hfi = Hf._inv_matrix
        for i in range(0, n_slcs):
            for j in range(0, n_tiles):
                H[i, j, :, :] = H00[i, j, :, :].dot(Hfi)

    return H


def get_canvas_bounds(imshape, n_slcs, n_tiles, H):
    """Get the corner coordinates of the transformed slices."""

    # compute the bounds for every slice in the transformed stack
    corner_coords = np.array([[0, 0], [0, imshape[1]],
                             [imshape[0], 0], imshape])  # [x, y]
    c = np.empty([n_slcs, n_tiles, 4, 2])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            ch = np.c_[corner_coords, np.ones(corner_coords.shape[0])]
            c[i, j, :, :] = ch.dot(H[i, j, :, :].T)[:, :2]

    # determine the bounds of the final canvas
    canvas_bounds = np.zeros([2, 2])
    canvas_bounds[0, :] = np.floor(np.amin(np.reshape(c, (-1, 2)), axis=0))
    canvas_bounds[1, :] = np.ceil(np.amax(np.reshape(c, (-1, 2)), axis=0))

    return canvas_bounds  # [[xmin,ymin],[xmax,ymax]]


def blend_tiles(outputdir, n_tiles, local_slcs, canvasshape, coordinates,
                H, distance_image, ds, order):
    """Warp and blend the tiles."""

    for i, tiles in local_slcs:
        # warp
        # imgslc in [y, x] order
        imgslc = np.zeros([canvasshape[0], canvasshape[1], n_tiles])
        disslc = np.zeros([canvasshape[0], canvasshape[1], n_tiles])
        for j, tile in enumerate(tiles):

            # tf_coord in [x, y] order
            tf_coord = coordinates.dot(np.linalg.inv(H[i, j, :, :]).T)[:, :2]
            newshape = (2, canvasshape[0], canvasshape[1])
            tf_coord = np.reshape(tf_coord.T, newshape)
            # tf_coord in [y, x] order
            tf_coord[[0, 1], :, :] = tf_coord[[1, 0], :, :]

            if ds > 1:
                im = rescale(tile, 1./ds)
            else:
                im = tile  # tile in [y, x] order
            # tfcoord should be [y, x] if im is [y, x]!
            imgslc[:, :, j] = map_coordinates(im, tf_coord, order=order)
            disslc[:, :, j] = map_coordinates(distance_image, tf_coord,
                                              order=order)

        # blend
        # imgcanvas in [y, x] order
        imgcanvas = np.zeros([canvasshape[0], canvasshape[1]])
        # nzcount is a map with number of contributing tiles
        nzcount = np.sum(disslc != 0, axis=2)
        distmp = disslc[nzcount > 0]
        s = np.tile(np.expand_dims(np.sum(distmp, axis=1), 1), (1, n_tiles))
        # w is a map of weights
        w = np.divide(distmp, s)
        imgcanvas[nzcount > 0] = np.sum(np.multiply(imgslc[nzcount > 0], w),
                                        axis=1)

        # save
        imsave(path.join(outputdir, str(i).zfill(4) + '.tif'), imgcanvas)

if __name__ == "__main__":
    main(sys.argv)
