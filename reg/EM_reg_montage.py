#!/usr/bin/env python

# def EM_reg_montage(n_tiles,offsets,overlap_fraction,subsamplefactor):
# TODO: generalize to other layouts
# TODO: check imsave datatype
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/tifs/0003* /Users/michielk/oxdata/P01/EM/M3/stitch_example/

from os import path, makedirs
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.feature import plot_matches
from skimage.transform import SimilarityTransform, ProjectiveTransform
from skimage.transform import warp, rescale
import math
from time import time
from matplotlib.pylab import c_, r_, s_
from scipy.optimize import minimize
from scipy.signal import gaussian
import pickle
import pprint
from skimage import img_as_float
from scipy.ndimage.interpolation import map_coordinates
from scipy.misc import imsave
from mpi4py import MPI

datadir = '/Users/michielk/oxdata/P01/EM/M3/stitch_example'
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/tifs'
outputdir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4'

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs'
outputdir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4'
upairsfile = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/unique_pairs_c1_d4.pickle'
n_tiles = 4
offsets = 1
downsample_factor = int(4)
subsample_factor = int(4)
n_kp=10000
# subsample_factor = int(4)
# n_kp=1000
overlap_fraction = [0.1, 0.1]  # [y,x]
pixsize = [1,1]  # [y,x]
connectivities = [['z', 0, 0],['z', 1, 1],['z', 2, 2],['z', 3, 3],
                  ['y', 0, 2],['y', 1, 3],
                  ['x', 0, 1],['x', 2, 3],
                  ['d1', 0, 3]]
pairsfile = path.join(datadir, 'pairs' + 
                      '_offsets' + str(offsets) + 
                      '_subsam' + str(subsample_factor) + '.pickle')

imgs = io.ImageCollection(path.join(datadir,'*.tif'))
n_slcs = len(imgs) / n_tiles
imgs = [imgs[(slc+1)*n_tiles-n_tiles:slc*n_tiles+4] 
        for slc in range(0, n_slcs)]

overlap_pixels = [int(math.ceil(d * overlap_fraction[i] * 1/subsample_factor)) 
                  for i,d in enumerate(imgs[0][0].shape)]
# TODO: catch different image shapes

# generate pairs (list of [[slcIm1,tileIm1], [slcIm2,tileIm2], 'type'])
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

pairstring = 'unique_pairs' + \
             '_c' + str(offsets) + \
             '_d' + str(downsample_factor)
pairfile = path.join(outputdir, pairstring + '.pickle')
pickle.dump(unique_pairs, open(pairfile, 'wb'))

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(unique_pairs)





pairs = load_pairs(datadir, offsets, subsample_factor, unique_pairs)
init_tfs = generate_init_tfs(pairs, n_slcs, n_tiles)
pc_factors = [0.2*math.pi, 0.001, 0.001]
init_tfs_pc = precondition_betas(init_tfs, pc_factors)

res = minimize(obj_fun_global, init_tfs_pc, 
               args=(pairs, pc_factors), method='L-BFGS-B',   # Nelder-Mead  # Powell
               options={'maxfun':100000, 'maxiter':1000, 'disp':True})
betas = decondition_betas(res.x, pc_factors)

betas = np.array(betas).reshape(n_slcs, n_tiles, len(pc_factors))
betasfile = path.join(datadir, 'betas' + 
                      '_o' + str(offsets) + 
                      '_s' + str(subsample_factor) + '.npy')
np.save(betasfile, betas)



retval = blend_tiles(imgs, betas, subsample_factor, order=1)

# TODO: bounded optimization???
# bnds = ((-0.02*math.pi,0.02*math.pi), (-200,200), (-200,200), 
#         (-0.02*math.pi,0.02*math.pi), (700,1100), (-200,200), 
#         (-0.02*math.pi,0.02*math.pi), (-200,200), (700,1100), 
#         (-0.02*math.pi,0.02*math.pi), (700,1100), (700,1100))


#####################
### function defs ###
#####################

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

def plot_pair_ransac(p, full_im1, full_im2, kp_im1, kp_im2, matches, inliers):
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
        print(plotdir)
    fig.savefig(path.join(plotdir, 'pair_s' + 
                          str(p[0][0]).zfill(4) + '-t' + 
                          str(p[0][1]) + '_s' +
                          str(p[1][0]).zfill(4) + '-t' + 
                          str(p[1][1]) + '.tif'))
    plt.close(fig)

def get_pairs(imgs, unique_pairs, offsets, subsample_factor, overlap_pixels, n_kp):
    """Create inlier keypoint pairs."""
    
    orb = ORB(n_keypoints=n_kp, fast_threshold=0.05)
    k = gaussian(offsets*2+1, 1, sym=True)
    tf = SimilarityTransform  # tf = RigidTransform
    tf0 = SimilarityTransform()
    # FIXME: is there no rigid model in scikit-image???
    # this is needed for input to RANSAC
    
    pairs = []
    init_tfs = np.empty([n_slcs, n_tiles, 3])
    for p in unique_pairs:
        pair_tstart = time()
        
        full_im1, full_im2 = subsample_images(p, imgs, subsample_factor)
        
        part_im1, part_im2 = select_imregions(p[2], full_im1, full_im2, overlap_pixels)
        keyp_im1, desc_im1 = get_keypoints(orb, part_im1)
        keyp_im2, desc_im2 = get_keypoints(orb, part_im2)
        keyp_im1, keyp_im2 = reset_imregions(p[2], keyp_im1, keyp_im2, 
                                                 overlap_pixels, 
                                                 full_im1.shape)
        
        matches = match_descriptors(desc_im1, desc_im2, cross_check=True)
        dst = keyp_im1[matches[:, 0]][:, ::-1]
        src = keyp_im2[matches[:, 1]][:, ::-1]
        model, inliers = ransac((src, dst), tf, min_samples=4, 
                                residual_threshold=2, max_trials=300)
        
        w = k[offsets - (p[1][0] - p[0][0])]
        
        pairs.append((p, src[inliers], dst[inliers], model, w))
        
        if (p[0][1] == 0) & (p[0][0] == p[1][0]):  # referenced to tile 0 within the same slice
            tf1 = tf0.__add__(model)
            itf = [math.acos(min(tf1.params[0,0], 1)),   # FIXME!!! with RigidTransform
                   tf1.params[0,2], tf1.params[1,2]]
            init_tfs[p[1][0],p[1][1],:] = np.array(itf)
        if (p[0][1] == p[1][1] == 0) & (p[1][0] - p[0][0] == 1):  # if [slcX,tile0] to [slcX-1,tile0]
            tf0 = tf0.__add__(model)
        
        plot_pair_ransac(p, full_im1, full_im2, 
                         keyp_im1, keyp_im2, matches, inliers)
        print('Pair done in: %.2f s' % (time() - pair_tstart,))
    
    return pairs, init_tfs


def load_pairs(datadir, offsets, subsample_factor, unique_pairs):
    """Load a previously generated set of pairs."""
    pairs = []
    for pid in range(0,len(unique_pairs)):
        pairfile = path.join(datadir, 'pairs' + 
                             '_o' + str(offsets) + 
                             '_s' + str(subsample_factor) + 
                             '_p' + str(pid).zfill(4) + '.pickle')
        p, src, dst, model, w = pickle.load(open(pairfile, 'rb'))
        pairs.append((p, src, dst, model, w))
    
    return pairs

def generate_init_tfs(pairs, n_slcs, n_tiles):
    """Find the transformation of each tile to tile[0,0]."""
    tf0 = SimilarityTransform()
    init_tfs = np.empty([n_slcs, n_tiles, 3])
    for pair in pairs:
        p, _, _, model, _ = pair
        if (p[0][1] == 0) & (p[0][0] == p[1][0]):  # referenced to tile 0 within the same slice
            tf1 = tf0.__add__(model)
            itf = [math.acos(min(tf1.params[0,0], 1)),   # FIXME!!! with RigidTransform
                   tf1.params[0,2], tf1.params[1,2]]
            init_tfs[p[1][0],p[1][1],:] = np.array(itf)
        if (p[0][1] == p[1][1] == 0) & (p[1][0] - p[0][0] == 1):  # if [slcX,tile0] to [slcX-1,tile0]
            tf0 = tf0.__add__(model)
    
    return init_tfs

def precondition_betas(betas, pc_factors):
    """Precondition the initial parameters for minimization."""
    betas = betas.reshape(betas.shape[0] * betas.shape[1], len(pc_factors))
    betas_pc = [list(b * pc_factors) for b in betas]
    
    return betas_pc

def decondition_betas(betas_pc, pc_factors):
    """Decondition the parameters after minimization."""
    l = len(pc_factors)
    betas_pc = np.array(betas_pc).reshape(betas_pc.size/l, l)
    betas_tmp = [list(i/pc_factors) for i in betas_pc]
    betas = [item for sublist in betas_tmp for item in sublist]
    
    return betas
    
def obj_fun_global(pars, pairs, pc):
    """."""
    # construct the transformation matrix for each slc/tile.
    H = np.empty([n_slcs, n_tiles, 3, 3])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            imno = i * n_tiles + j
            if i+j == 0:
                H[i,j,:,:] = np.identity(3)
            else:
                theta = pars[(imno+1)*3-3] / pc[0]
                tx = pars[(imno+1)*3-2] / pc[1]
                ty = pars[(imno+1)*3-1] / pc[2]
                H[i,j,:,:] = np.array([[math.cos(theta), -math.sin(theta), tx],
                                       [math.sin(theta),  math.cos(theta), ty],
                                       [0,0,1]])
    
    wses = np.empty([0,2])
    for i, (p,s,d,_,w) in enumerate(pairs):
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
    sse = sum(sum(wses))
    return sse


def create_distance_image(imshape):
    """Create an image coding for every pixel the distance from the edge."""
    # TODO: get a Gaussian weighting on the image???
    xd = np.linspace(-imshape[1]/2, imshape[1]/2, imshape[1])
    yd = np.linspace(-imshape[0]/2, imshape[0]/2, imshape[0])
    Xd,Yd = np.meshgrid(xd, yd, sparse=True)
    distance_image = np.minimum(-(np.abs(Xd) - imshape[1]/2), 
                                -(np.abs(Yd) - imshape[0]/2))
    
    return distance_image

def tfs_to_H(tfs):
    """Convert a N*Mx3 array of [th,tx,ty] to NxM 3x3 tf matrices."""
    H = np.empty([n_slcs, n_tiles, 3, 3])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            imno = i * n_tiles + j
            if i+j == 0:
                H[i,j,:,:] = np.identity(3)
            else:
                theta = tfs[(imno+1)*3-3]
                tx = tfs[(imno+1)*3-2]
                ty = tfs[(imno+1)*3-1]
                H[i,j,:,:] = np.array([[math.cos(theta), -math.sin(theta), tx],
                              [math.sin(theta),  math.cos(theta), ty],
                              [0,0,1]])
            
#             if j == 0:
#                 H[i,j,:,:] = np.identity(3)
#             else:
#                 if j==1:
#                     theta = 0; tx = -100; ty = 900
#                 elif j==2:
#                     theta = 0; tx = 900; ty = -100
#                 elif j==3:
#                     theta = 0; tx = 800; ty = 800
#                 H[i,j,:,:] = np.array([[math.cos(theta), -math.sin(theta), tx],
#                                        [math.sin(theta),  math.cos(theta), ty],
#                                        [0,0,1]])
    
    return H

def get_canvas_bounds(imshape, n_slcs, n_tiles, H):
    """Get the corner coordinates of the transformed slices."""
    # compute the bounds for every slice in the transformed stack
    corner_coords = np.array([[0,0],[0,imshape[1]],[imshape[0],0],imshape])  # [x,y]
    c = np.empty([n_slcs, n_tiles, 4, 2])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            imno = i * n_tiles + j
            ch = c_[corner_coords, np.ones(corner_coords.shape[0])]
            c[i,j,:,:] = ch.dot(H[i,j,:,:].T)[:,:2]
    
    # determine the bounds of the final canvas
    canvas_bounds = np.zeros([2,2])
    canvas_bounds[0,:] = np.floor(np.amin(np.reshape(c, (-1,2)), axis=0))
    canvas_bounds[1,:] = np.ceil(np.amax(np.reshape(c, (-1,2)), axis=0))
    
    return canvas_bounds  # [[xmin,ymin],[xmax,ymax]]

def blend_tiles(imgs, tfs, subsample_factor=1, order=2):
    """Warp and blend the tiles."""
    
    imshape = [s/subsample_factor for s in imgs[0][0].shape]
    distance_image = create_distance_image(imshape)
    
    H = tfs_to_H(tfs)
    canvas_bounds = get_canvas_bounds(imshape, n_slcs, n_tiles, H)  ## [[xmin,ymin],[xmax,ymax]]
    
    # create the canvas coordinate list
    x = np.linspace(canvas_bounds[0,0], canvas_bounds[1,0], 
                    canvas_bounds[1,0]-canvas_bounds[0,0])
    y = np.linspace(canvas_bounds[0,1], canvas_bounds[1,1], 
                    canvas_bounds[1,1]-canvas_bounds[0,1])
    X, Y = np.meshgrid(x, y)  # [y,x] shape
    coordinates = c_[np.ndarray.flatten(X), np.ndarray.flatten(Y),
                     np.ones_like(np.ndarray.flatten(X))]  # [x,y]
    
    # warp and blend for each slice
    for i,tiles in enumerate(imgs):
        # warp
        imgslc = np.zeros([X.shape[0], X.shape[1], n_tiles])  # [y,x]
        disslc = np.zeros([X.shape[0], X.shape[1], n_tiles])
        for j,tile in enumerate(tiles):
            
            tf_coord = coordinates.dot(np.linalg.inv(H[i,j,:,:]).T)[:,:2]  # [x,y]
            tf_coord = np.reshape(tf_coord.T, (2, X.shape[0], X.shape[1]))
            tf_coord[[0,1],:,:] = tf_coord[[1,0],:,:]  # [y,x]
            
            if subsample_factor > 1:
                im = rescale(imgs[i][j], 1./subsample_factor)
            else:
                im = imgs[i][j]  # [y,x]
            imgslc[:,:,j] = map_coordinates(im, tf_coord, order=order)  # tfcoord should be [y,x] if im is [y,x]
            disslc[:,:,j] = map_coordinates(distance_image, tf_coord, order=order)
        
        # blend
        imgcanvas = np.zeros([X.shape[0], X.shape[1]])  # [y,x]
        nzcount = np.sum(disslc != 0, axis=2)
        distmp = disslc[nzcount>0]
        s = np.tile(np.expand_dims(np.sum(distmp, axis=1), 1), (1,n_tiles))
        w = np.divide(distmp, s)
        imgcanvas[nzcount>0] = np.sum(np.multiply(imgslc[nzcount>0], w), axis=1)
        
        # save
        imsave(path.join(datadir, 'imgcanvas' + str(i) + '.tif'), imgcanvas)
    
    return 0






