scriptdir=/Users/michielk/workspace/EM
datadir=/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs
reference_name=0250_m000.tif
mkdir -p $datadir/reg/trans

sed "s?SOURCE_DIR?$datadir?;\
    s?TARGET_DIR?$datadir/reg?;\
    s?REFNAME?$reference_name?g" \
    $scriptdir/EM_register.py \
    > $datadir/EM_register.py

ImageJ --headless $datadir/EM_register.py


mpiexec -n 8 python /Users/michielk/workspace/EM/EM_series2stack.py \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg' \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg/tifs.nii.gz' -m -o -e 0.0073 0.0073 0.05



# TODO: 
# check job getpairs
# submit and monitor optimize
# submit blend
# evaluate

# evaluate result biexponential fit
# send results

# get the pairs and write to file



qsubfile=$datadir/EM_reg_optimize_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python $scriptdir/EM_reg_optimize.py \
'$datadir/tifs' -o '$datadir/reg_d4' \
-u '$datadir/reg_d4/unique_pairs_c1_d4.pickle' \
-t 4 -c 1 -d 4 -a init" >> $qsubfile
qsub -q develq $qsubfile

mpiexec -n 8 python /Users/michielk/workspace/EM/EM_reg_blend.py \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs' \
-t 4 -c 1 -d 4 -i 1 -m


rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4/b* $local_datadir/reg_d4/




python /Users/michielk/workspace/EM/EM_reg_optimize.py \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4' \
-t 4 -c 1 -d 4 -i 10
python /Users/michielk/workspace/EM/EM_reg_blend.py \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs' \
-t 4 -c 1 -d 4 -i 1




mpiexec -n 8 python /Users/michielk/workspace/EM/EM_downsample.py \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/00tifs' \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/00tifs_ds' \
-d 4 -m
mpiexec -n 8 python /Users/michielk/workspace/EM/EM_series2stack.py \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/00tifs_ds' \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/00tifs_ds.h5' \
-f 'reg_ds' -m -o -e 0.0584 0.0584 0.05


python /Users/michielk/workspace/EM/EM_series2stack.py \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/00tifs_ds' \
'/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/00tifs_ds.h5' \
-f 'reg_ds' -o -e 0.0584 0.0584 0.05



mpiexec -n 2 python /Users/michielk/workspace/EM/EM_reg_optimize.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack/reg' \
-t 4 -c 1 -d 4 -i 10 -m

mpiexec -n 2 python /Users/michielk/workspace/EM/EM_reg_optimize.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example/reg' \
-t 4 -c 1 -d 4 -i 10 -m



qsubfile=$datadir/EM_reg_getpairs_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=4:ppn=16" >> $qsubfile
echo "#PBS -l walltime=01:00:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/EM_reg_getpairs.py \
'$datadir/tifs' -o '$datadir/reg_d4' \
-t 4 -c 1 -d 4 -k 1000 -f 0.1 0.1 -m -n 100 -r 1" >> $qsubfile
qsub $qsubfile

qsubfile=$datadir/EM_reg_optimize_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "python $scriptdir/EM_reg_optimize.py \
'$datadir/tifs' -o '$datadir/reg_d4' \
-t 4 -c 1 -d 4 -i 10" >> $qsubfile
qsub -q develq $qsubfile

python /Users/michielk/workspace/EM/EM_reg_optimize.py \
'/Users/michielk/oxdata/P01/EM/M3/tifs' \
-o '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4' \
-u '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/unique_pairs_c1_d4.pickle' \
-t 4 -c 1 -d 4 -i 10

mpiexec -n 6 python /Users/michielk/workspace/EM/EM_reg_optimize.py \
'/Users/michielk/oxdata/P01/EM/M3/tifs' \
-o '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4' \
-u '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/reg_d4/unique_pairs_c1_d4.pickle' \
-t 4 -c 1 -d 4 -i 10 -m



qsubfile=$datadir/EM_reg_blend_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/EM_reg_blend.py \
'$datadir/tifs' -o '$datadir/reg_d4' \
-t 4 -c 1 -d 4 -i 1 -m" >> $qsubfile
qsub $qsubfile




mpiexec -n 4 python /Users/michielk/workspace/EM/EM_reg_getpairs.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example/reg' \
-t 4 -c 1 -d 1 -k 10000 -f 0.1 0.1 -m -p -n 100 -r 2

mpiexec -n 4 python /Users/michielk/workspace/EM/EM_reg_getpairs.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack/reg' \
-t 4 -c 1 -d 1 -k 1000 -f 0.1 0.1 -m -p -n 100 -r 2

mpiexec -n 6 python /Users/michielk/workspace/EM/EM_reg_getpairs.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack/reg' \
-t 4 -c 1 -d 4 -k 1000 -f 0.1 0.1 -m -p -n 100 -r 1
mpiexec -n 6 python /Users/michielk/workspace/EM/EM_reg_optimize.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack/reg' \
-t 4 -c 1 -d 4 -i 100 -m
# mpiexec -n 1 
python /Users/michielk/workspace/EM/EM_reg_blend.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example_midstack/reg' \
-t 4 -c 1 -d 4 -i 1




python /Users/michielk/workspace/EM/EM_reg_optimize.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example/reg' \
-t 4 -c 1 -d 1 -i 10

mpiexec -n 1 python /Users/michielk/workspace/EM/EM_reg_blend.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example' \
-o '/Users/michielk/oxdata/P01/EM/M3/stitch_example/reg' \
-t 4 -c 1 -d 1 -i 1 -m

mpiexec -n 5 python /Users/michielk/workspace/EM/EM_downsample.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example/reg' \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example/reg_ds' \
-d 10 -m

mpiexec -n 5 python /Users/michielk/workspace/EM/EM_series2stack.py \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example/reg_ds' \
'/Users/michielk/oxdata/P01/EM/M3/stitch_example/reg_ds.h5' \
-f 'reg_ds' -m -o -e 0.073 0.073 0.05



datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/tifs'
datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/stitch_example_midstack'
n_tiles = 4
offsets = 1
subsample_factor = int(1)
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

orb = ORB(n_keypoints=10000, fast_threshold=0.05)
k = gaussian(offsets*2+1, 1, sym=True)
pid=6
pair = get_pair(datadir, imgs, unique_pairs[pid], pid, offsets, 1, overlap_fraction, orb, k, 0, res_th=10, num_inliers=100)

rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/tifs/p* $local_datadir

rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4/p* $local_datadir/reg_d4





# imreg_dft.imreg.similarity(im0, im1, numiter=1, order=3, constraints=None, filter_pcorr=0, exponent='inf', reports=None)
# 
# imreg_dft.imreg.transform_img(img, scale=1.0, angle=0.0, tvec=(0, 0), bgval=None, order=1)
# imreg_dft.imreg.transform_img_dict(img, tdict, bgval=None, order=1, invert=False)
# 
# imreg_dft.imreg.imshow(im0, im1, im2, cmap=None, fig=None, **kwargs)

import os

import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt

import imreg_dft as ird
# import imreg_dft.utils as utils
import imreg_dft.tiles as tiles

datadir = '/Users/michielk/oxdata/P01/EM/M3/stitch_example'
# dst = sp.misc.imread(os.path.join(datadir, "0000_m000.tif"), True)
# src = sp.misc.imread(os.path.join(datadir, "0001_m000.tif"), True)
full_im1, full_im2 = subsample_images(p, imgs, subsample_factor)
dst, src = select_imregions(p[2], full_im1, full_im2, overlap_pixels)

constraints = {'angle': [0,5], 'scale': [1,0]}
result = ird.similarity(dst, src, numiter=3, constraints=constraints)
ird.imshow(dst, src, result['timg'])
plt.show()



# subpixel
ird sample1.png sample3.png --resample 3 --iter 4 --lowpass 0.9,1.1 --extend 10 --print-result










from numpy import mean, absolute

def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)








datadir = '/Users/michielk/oxdata/P01/EM/M3/stitch_example'

from os import path
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
from matplotlib.pylab import c_, r_
from scipy.optimize import minimize

# TODO: parallelize with mpi4py
# TODO: work in um, not pixels???
# TODO: generalize to other layouts
# TODO: change pairs 0-7 to slc0->0-3 slc1->0-3 (slc2->0-3)
# TODO make into callable function

# get a list of all images
all_imgs = io.ImageCollection(path.join(datadir,'*.tif'))

n_tiles = 4
n_slcs = len(all_imgs) / n_tiles

# specify the layout of the data; this is for two slices
pairs = [[0,4],[1,5],[2,6],[3,7],
         [0,2],[1,3],[4,6],[5,7], [0,6], [1,7], 
         [0,1],[2,3],[4,5],[6,7], [0,5], [2,7],
         [0,3],[4,7],[0,7],[4,3],
         [1,2],[5,6],[1,6],[5,2]]
# indexes in pairs of different pair-types
zpairs = [0,1,2,3]
ypairs = [4,5,6,7,8,9]
xpairs = [10,11,12,13,14,15]
d1pairs = [16,17,18,19]  # topleft-rightbottom
d2pairs = [20,21,22,23]  # topright-leftbottom # FIXME: these dont have much overlap
zero_pairs = [10,4,16,0,14,8,18]

subsample_factor = int(4)
overlap_fraction = [0.1, 0.1]  # [y,x]
overlap_pixels = [int(math.ceil(d * overlap_fraction[i] * 1/subsample_factor)) 
                  for i,d in enumerate(all_imgs[0].shape)]

orb = ORB(n_keypoints=1000, fast_threshold=0.05)

tf = SimilarityTransform
tf = RigidTransform
# FIXME: is there no rigid model in scikit-image??? # this is needed for input to RANSAC

all_pairs = []
init_tfs =  []
for slc in range(0, n_slcs - 1):
    slc_tstart = time()
    
    imgs = all_imgs[(slc+1)*4-4:slc*4+8]
    
    for i,pair in enumerate(pairs):
        pair_tstart = time()
        
        # subsample images
        if subsample_factor > 1:
            full_im1 = rescale(imgs[pair[0]], 1./subsample_factor)
            full_im2 = rescale(imgs[pair[1]], 1./subsample_factor)
        else:
            full_im1 = imgs[pair[0]]
            full_im2 = imgs[pair[1]]
        
        # select image regions to extract keypoints from
        if i in zpairs:
            im1 = full_im1
            im2 = full_im2
        elif i in ypairs:
            y1 = full_im1.shape[0] - overlap_pixels[0]
            y2 = overlap_pixels[0]
            im1 = full_im1[y1:,:]
            im2 = full_im2[:y2,:]
        elif i in xpairs:
            x1 = full_im1.shape[1] - overlap_pixels[1]
            x2 = overlap_pixels[1]
            im1 = full_im1[:,x1:]
            im2 = full_im2[:,:x2]
        elif i in d1pairs:
            x1 = full_im1.shape[1] - 2 * overlap_pixels[1]
            y1 = full_im1.shape[0] - 2 * overlap_pixels[0]
            x2 = 2 * overlap_pixels[1]
            y2 = 2 * overlap_pixels[0]
            im1 = full_im1[y1:,x1:]
            im2 = full_im2[:y2,:x2]
        elif i in d2pairs:
            x1 = full_im1.shape[1] - 2 * overlap_pixels[1]
            y1 = 2 * overlap_pixels[0]
            x2 = 2 * overlap_pixels[1]
            y2 = full_im2.shape[0] - 2 * overlap_pixels[0]
            im1 = full_im1[:y1,x1:]
            im2 = full_im2[y2:,:x2]
        
        # get the matching keypoints
        orb.detect_and_extract(im1)
        kp_im1 = orb.keypoints
        ds_im1 = orb.descriptors
        orb.detect_and_extract(im2)
        kp_im2 = orb.keypoints
        ds_im2 = orb.descriptors
        
        # transform keypoints back to full image space
        if i in zpairs:
            pass
        elif i in ypairs:
            kp_im1[:,0] += full_im1.shape[0] - overlap_pixels[0]
        elif i in xpairs:
            kp_im1[:,1] += full_im1.shape[1] - overlap_pixels[1]
        elif i in d1pairs:
            kp_im1[:,0] += full_im1.shape[0] - 2 * overlap_pixels[0]
            kp_im1[:,1] += full_im1.shape[1] - 2 * overlap_pixels[1]
        elif i in d2pairs:
            kp_im1[:,0] += full_im1.shape[0] - 2 * overlap_pixels[0]
            kp_im2[:,1] += full_im1.shape[1] - 2 * overlap_pixels[1]
        
        matches = match_descriptors(ds_im1, ds_im2, cross_check=True)
        src = kp_im2[matches[:, 1]][:, ::-1]
        dst = kp_im1[matches[:, 0]][:, ::-1]
        
        # apply ransac to determine inliers and keep only those
        model, inliers = ransac((src, dst), tf,
                                min_samples=4, 
                                residual_threshold=2, 
                                max_trials=300)
        
        all_pairs.append((pair, src[inliers], dst[inliers], model))  # TODO: write results to a logfile
        
        if i in zero_pairs: # select only pairs [0,x] FIXME: works only for closest slices
            if model.params[0,0] > 1:  # FIXME!!! with RigidTransform
                model.params[0,0] = 1
            ps = [math.acos(model.params[0,0]), model.params[0,2], model.params[1,2]]
            init_tfs.extend(ps)
            
        # plot results
#         fig, (ax1,ax2) = plt.subplots(2,1)
#         plot_matches(ax1, full_im1, full_im2, kp_im1, kp_im2, 
#                      matches, only_matches=True)
#         ax1.axis('off')
#         plot_matches(ax2, full_im1, full_im2, kp_im1, kp_im2, 
#                      matches[inliers], only_matches=True)
#         ax2.axis('off')
#         fig.savefig(path.join(datadir,'pair' + str(i) + '.tif'))
#         fig.clf()
        
        print('Pair done in: %.2f s' % (time() - pair_tstart,))
    print('---- Slice done in: %.2f s' % (time() - slc_tstart,))
#     return all_pairs, init_tfs

res = minimize(obj_fun_global, init_tfs, 
               args=(all_pairs), method='Nelder-Mead', 
               options={'maxfev':100000, 'maxiter':100000, 'disp':True})


# loop over consecutive image pairs


# apply transforms to images
np.set_printoptions(suppress=True)
for i,img in enumerate(imgs):
    if i == 0:
        pass
    else:
        par = res.x[i*3-3:i*3]
        H = np.array([[math.cos(par[0]), -math.sin(par[0]), par[1]],
                      [math.sin(par[0]),  math.cos(par[0]), par[2]],
                      [0,0,1]])
        tf = SimilarityTransform(H)
        print(np.around(tf.params, decimals=2))
        # get the extreme points for each image to calculate the output image size?
        # 
        im000 = warp(im, H, output_shape=...)



def obj_fun_global(pars, all_pairs):
    
#     pars = init_tfs
    H = np.empty([n_slcs, n_tiles, 3, 3])
    for i in range(0, n_slcs):
        for j in range(0, n_tiles):
            imno = i * n_tiles + j
            if i+j == 0:
                H[i,j,:,:] = np.identity(3)
            else:
                theta = pars[imno*3-3]
                tx = pars[imno*3-2]
                ty = pars[imno*3-1]
                H[i,j,:,:] = np.array([[math.cos(theta), -math.sin(theta), tx],
                                       [math.sin(theta),  math.cos(theta), ty],
                                       [0,0,1]])
    
    sse = 0
    all_d = np.empty([0,2])
    all_s = np.empty([0,2])
    for i,slcpairs in enumerate(all_pairs):
        for j, (pair,s,d,model) in enumerate(slcpairs):
            # homogenize pointsets
            d = c_[d, np.ones(d.shape[0])]
            s = c_[s, np.ones(s.shape[0])]
            # transform d/s points to image000 space
            d = d.dot(H[i][j].T)[:,:2]
            s = s.dot(H[i][j].T)[:,:2]
            all_d = r_[all_d, d]
            all_s = r_[all_s, s]
    # square and sum
    ## TODO: weight each point-pair by importance / distance???
    sse += sum(sum( (all_d - all_s)**2 ))
    
    return sse


# # initialize on individual transforms to im000
# init_tfs =  []
# for i in [10,4,14,0,14,8,18]:
#     p,s,d,m = all_pairs[i]
#     if m.params[0,0] > 1:  # FIXME!!! with RigidTransform
#         m.params[0,0] = 1
#     init_tfs.append(math.acos(m.params[0,0]))
#     init_tfs.append(m.params[0,2])
#     init_tfs.append(m.params[1,2])
# 
# def obj_fun_global(pars, all_pairs):
#     # 4 image per slice; 2 slices; all but one image moving; estimate 7 transforms
#     theta1, tx1, ty1, \
#     theta2, tx2, ty2, \
#     theta3, tx3, ty3, \
#     theta4, tx4, ty4, \
#     theta5, tx5, ty5, \
#     theta6, tx6, ty6, \
#     theta7, tx7, ty7 = pars
#     
#     H0 = np.array([[1,0,0],
#                    [0,1,0],
#                    [0,0,1]])
#     H1 = np.array([[math.cos(theta1), -math.sin(theta1), tx1],
#                    [math.sin(theta1),  math.cos(theta1), ty1],
#                    [0,0,1]])
#     H2 = np.array([[math.cos(theta2), -math.sin(theta2), tx2],
#                    [math.sin(theta2),  math.cos(theta2), ty2],
#                    [0,0,1]])
#     H3 = np.array([[math.cos(theta3), -math.sin(theta3), tx3],
#                    [math.sin(theta3),  math.cos(theta3), ty3],
#                    [0,0,1]])
#     H4 = np.array([[math.cos(theta4), -math.sin(theta4), tx4],
#                    [math.sin(theta4),  math.cos(theta4), ty4],
#                    [0,0,1]])
#     H5 = np.array([[math.cos(theta5), -math.sin(theta5), tx5],
#                    [math.sin(theta5),  math.cos(theta5), ty5],
#                    [0,0,1]])
#     H6 = np.array([[math.cos(theta6), -math.sin(theta6), tx6],
#                    [math.sin(theta6),  math.cos(theta6), ty6],
#                    [0,0,1]])
#     H7 = np.array([[math.cos(theta7), -math.sin(theta7), tx7],
#                    [math.sin(theta7),  math.cos(theta7), ty7],
#                    [0,0,1]])
#     H = [H0,H1,H2,H3,H4,H5,H6,H7]
#     
#     sse = 0
#     for pair,s,d,model in all_pairs:
#         # homogenize pointsets
#         d = c_[d, np.ones(d.shape[0])]
#         s = c_[s, np.ones(s.shape[0])]
#         # transform d/s points to image000 space
#         d = d.dot(H[pair[0]].T)[:,:2]
#         s = s.dot(H[pair[1]].T)[:,:2]
#         # square and sum
#         sse += sum(sum( (d - s)**2 ))
#     # square and sum
#     sse += sum(sum( (d - s)**2 ))
#     
#     return sse








#     pwpars = []
#     Hps = []
#     for i,(src,dst) in enumerate(all_pairs):
#         # pairwise transformations
#         pwpar = minimize(obj_fun, [0,0,0], 
#                        args=(src,dst), method='Nelder-Mead')
#         pwpars.append(pwpar)
#         Hp = np.array([[math.cos(pwpar[0]), -math.sin(pwpar[0]), pwpar[1]],
#                        [math.sin(pwpar[0]),  math.cos(pwpar[0]), pwpar[2]],
#                        [0,0,1]])
#         Hps.append(Hp)





# from: http://stackoverflow.com/questions/26574303/estimate-euclidean-transformation-with-python
# from matplotlib.pylab import *
# from scipy.optimize import *
# def obj_fun(pars,x,src):
#     theta, tx, ty = pars
#     H = array([[cos(theta), -sin(theta), tx],\
#          [sin(theta), cos(theta), ty],
#          [0,0,1]])
#     src1 = c_[src,ones(src.shape[0])]
#     return sum( (x - src1.dot(H.T)[:,:2])**2 )
# def apply_transform(pars, src):
#     theta, tx, ty = pars
#     H = array([[cos(theta), -sin(theta), tx],\
#          [sin(theta), cos(theta), ty],
#          [0,0,1]])
#     src1 = c_[src,ones(src.shape[0])]
#     return src1.dot(H.T)[:,:2]
# 
# for pair,dst,src in all_pairs:
#     res0 = minimize(obj_fun,[0,0,0],args=(dst,src), method='Nelder-Mead')
#     print(res0)













# from scikit-image
# import six
# import math
# import warnings
# import numpy as np
# from scipy import spatial
# from scipy import ndimage as ndi
# 
# from .._shared.utils import get_bound_method_class, safe_as_int
# from ..util import img_as_float
# from ._warps_cy import _warp_fast

def _center_and_normalize_points(points):
    """Center and normalize image points.
    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.
    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.
    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).
    Parameters
    ----------
    points : (N, 2) array
        The coordinates of the image points.
    Returns
    -------
    matrix : (3, 3) array
        The transformation matrix to obtain the new points.
    new_points : (N, 2) array
        The transformed image points.
    """
    
    centroid = np.mean(points, axis=0)
    
    rms = math.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])
    
    norm_factor = math.sqrt(2) / rms
    
    matrix = np.array([[norm_factor, 0, -norm_factor * centroid[0]],
                       [0, norm_factor, -norm_factor * centroid[1]],
                       [0, 0, 1]])
    
    pointsh = np.row_stack([points.T, np.ones((points.shape[0]),)])
    
    new_pointsh = np.dot(matrix, pointsh).T
    
    new_points = new_pointsh[:, :2]
    new_points[:, 0] /= new_pointsh[:, 2]
    new_points[:, 1] /= new_pointsh[:, 2]
    
    return matrix, new_points

class RigidTransform(ProjectiveTransform):
    """2D similarity transformation of the form:
    ..:math:
        X = a0 * x - b0 * y + a1 =
          = m * x * cos(rotation) - m * y * sin(rotation) + a1
        Y = b0 * x + a0 * y + b1 =
          = m * x * sin(rotation) + m * y * cos(rotation) + b1
    where ``m`` is a zoom factor and the homogeneous transformation matrix is::
        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]
    Parameters
    ----------
    matrix : (3, 3) array, optional
        Homogeneous transformation matrix.
    scale : float, optional
        Scale factor.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians.
    translation : (tx, ty) as array, list or tuple, optional
        x, y translation parameters.
    Attributes
    ----------
    params : (3, 3) array
        Homogeneous transformation matrix.
    """
    
    def __init__(self, matrix=None, scale=None, rotation=None,
                 translation=None):
        params = any(param is not None
                     for param in (scale, rotation, translation))
        
        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params:
            if scale is None:
                scale = 1
            if rotation is None:
                rotation = 0
            if translation is None:
                translation = (0, 0)
                
            self.params = np.array([
                [math.cos(rotation), - math.sin(rotation), 0],
                [math.sin(rotation),   math.cos(rotation), 0],
                [                 0,                    0, 1]
            ])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)
    
    def estimate(self, src, dst):
        """Set the transformation matrix with the explicit parameters.
        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.
        Number of source and destination coordinates must match.
        The transformation is defined as::
            Xi = ai0 * xi - bi0 * yi + ai1
            Yi = bi0 * xi + ai0 * yi + bi1
        These equations can be transformed to the following form::
            0 = ai0 * xi - bi0 * yi + ai1 - Xi
            0 = bi0 * xi + ai0 * yi + bi1 - Yi
        which exist for each set of corresponding points, so we have a set of
        N * 2i equations. The coefficients appear linearly so we can write
        A x = 0, where::
            A   = [[x1 1 -y1 0 -X1]
                   [y1 0  x1 1 -Y1]
                   [x2 1 -y2 0 -X2]
                   [y2 0  x2 1 -Y2]
                    ...
                    ...
                  ]
            x.T = [a0 a1 b0 b1 c3]
        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.
        Parameters
        ----------
        src : (N, 2i) array
            Source coordinates.
        dst : (N, 2i) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        
        try:
            src_matrix, src = _center_and_normalize_points(src)
            dst_matrix, dst = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = np.nan * np.empty((3, 3))
            return False
        
        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]
        
        # params: a0, a1, b0, b1
        A = np.zeros((rows * 2, 5))
        A[:rows, 0] = xs
        A[:rows, 2] = - ys
        A[:rows, 1] = 1
        A[rows:, 2] = xs
        A[rows:, 0] = ys
        A[rows:, 3] = 1
        A[:rows, 4] = xd
        A[rows:, 4] = yd
        
        _, _, V = np.linalg.svd(A)
        
        # solution is right singular vector that corresponds to smallest
        # singular value
        a0, a1, b0, b1 = - V[-1, :-1] / V[-1, -1]
        
        S = np.array([[a0, -b0, a1],
                      [b0,  a0, b1],
                      [ 0,   0,  1]])
        
        # De-center and de-normalize
        S = np.dot(np.linalg.inv(dst_matrix), np.dot(S, src_matrix))
        
        self.params = S
        
        return True
    
    @property
    def scale(self):
        if abs(math.cos(self.rotation)) < np.spacing(1):
            # sin(self.rotation) == 1
            scale = self.params[1, 0]
        else:
            scale = self.params[0, 0] / math.cos(self.rotation)
        return scale
    
    @property
    def rotation(self):
        return math.atan2(self.params[1, 0], self.params[1, 1])
    
    @property
    def translation(self):
        return self.params[0:2, 2]












# # opencv basics
# import cv2
# import numpy as np
# 
# img1 = cv2.imread('ml.png')
# img2 = cv2.imread('opencv_logo.jpg')
# dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 
# 
# # scaling
# img = cv2.imread('messi5.jpg')
# res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
# #OR
# height, width = img.shape[:2]
# res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
# 
# # translation
# img = cv2.imread('messi5.jpg',0)
# rows,cols = img.shape
# M = np.float32([[1,0,100],[0,1,50]])
# dst = cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow('img',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 
# # rotation
# img = cv2.imread('messi5.jpg',0)
# rows,cols = img.shape
# M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
# dst = cv2.warpAffine(img,M,(cols,rows))
# st = cv2.Stitcher
# 
# # affine
# img = cv2.imread('drawing.png')
# rows,cols,ch = img.shape
# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])
# M = cv2.getAffineTransform(pts1,pts2)
# dst = cv2.warpAffine(img,M,(cols,rows))
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()
# 
# # ransac
# H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
# 
# 
# cv2.estimateRigidTransform



import cv2
import numpy as np,sys

A = cv2.imread('apple.jpg')
B = cv2.imread('orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in xrange(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)





