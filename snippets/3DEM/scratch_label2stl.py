# import os
# import sys
# from argparse import ArgumentParser
# import errno
# import h5py
# import numpy as np
# import vtk
# from skimage.morphology import remove_small_objects
# from skimage.measure import label
# from scipy.ndimage.interpolation import shift
# from scipy.ndimage.morphology import binary_fill_holes
#
# datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU"
# x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
# dataset = 'm000'
# fieldnamein = 'stack'
# nzfills = 5
# zyxOffset = [30,1000,1000]
# labelimages = ['_probs_ws_MAfilled', '_probs_ws_PA']
# dataset = dataset + '_' + str(x).zfill(nzfills) + '-' + str(X).zfill(nzfills) + \
#                     '_' + str(y).zfill(nzfills) + '-' + str(Y).zfill(nzfills) + \
#                     '_' + str(z).zfill(nzfills) + '-' + str(Z).zfill(nzfills)
#
# mask, _ = loadh5(datadir, dataset + labelimages[0] + '.h5', fieldnamein)
# mask = np.ones_like(mask, dtype='bool')
# ECSmask = np.zeros_like(mask, dtype='bool')
#
# l=labelimages[0]
# compdict = {}
# labeldata, elsize = loadh5(datadir, dataset + l + '.h5', fieldnamein)
# labeldata[~mask] = 0
# if 'PA' in l:
#     compdict['MM'] = np.unique(labeldata[np.logical_and(labeldata>1000, labeldata<2000)])
#     compdict['UA'] = np.unique(labeldata[np.logical_and(labeldata>2000, labeldata<6000)])
#     compdict['GB'] = np.unique(labeldata[np.logical_and(labeldata>6000, labeldata<7000)])
#     compdict['GP'] = np.unique(labeldata[np.logical_and(labeldata>7000, labeldata<8000)])
# else:
#     compdict['MA'] = np.unique(labeldata)
#
# labeldata = remove_small_objects(labeldata, 100)
#
# labels2meshes_vtk(datadir, compdict, np.transpose(labeldata), spacing=np.absolute(elsize)[::-1], offset=zyxOffset[::-1])
# ECSmask[labeldata>0] = True


### ARC ###
source ~/.bashrc
scriptdir="${HOME}/workspace/EM"
oddir="${DATA}/EM/M3/20Mar15/montage/Montage_"
datadir="${DATA}/EM/M3/M3_S1_GNU" && cd ${datadir}
dataset='m000'
module load hdf5-parallel/1.8.14_mvapich2_intel #module load hdf5-parallel/1.8.14_mvapich2
module load mvapich2/2.0.1__intel-2015
module load mpi4py/1.3.1
module load python/2.7__gcc-4.8
module load vtk/5.8.0  #vtk/5.10.1
source activate root
python

x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
qsubfile=$datadir/EM_l2s.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_l2s" >> $qsubfile
echo "export PATH=${HOME}/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate root" >> $qsubfile
echo "echo `which python`" >> $qsubfile
echo "python ${scriptdir}/mesh/label2stl.py ${datadir} ${dataset} \
-L '_MA_sMAkn_sUAkn_ws_filled' '_PA' -f '/stack' -n 5 -o ${z} ${y} ${x} \
-x ${x} -X ${X} -y ${y} -Y ${Y} -z ${z} -Z ${Z}" >> $qsubfile
sbatch -p devel $qsubfile

### local ###
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
x=1000; X=1500; y=1000; Y=1500; z=200; Z=300;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`

# label2stl.py latest
python $scriptdir/mesh/label2stl.py $datadir/$datastem $dataset \
-L '_MA_sMAkn_sUAkn_ws_filled' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z
python $scriptdir/mesh/label2stl.py $datadir/$datastem $dataset \
-L '_PA' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

for comp in MM NN GP UA; do
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf_PAenforceECS ${comp} -L ${comp} -s 0.5 10 -d 0.2  -e 0.01
done
for comp in ECS; do
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf ${comp} -L ${comp} -d 0.02
done

for ob in bpy.data.objects:
    ob.hide = True
