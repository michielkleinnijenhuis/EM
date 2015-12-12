rsync -avz /Users/michielk/oxdata/originaldata/P01/EM/M2/Brain/I/ ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/
rsync -avz /Users/michielk/oxdata/P01/EM/M2/I/ ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/
# rsync -avz /Users/michielk/oxdata/P01/EM/M2/J/*_slicvoxels.h5 ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/J/
rsync -avz /Users/michielk/oxdata/P01/EM/M2/I/test_data_large.h5 ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/
rsync -avz /Users/michielk/oxdata/P01/EM/M2/I/test_data_large_Probabilities.h5 ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/
rsync -avz /Users/michielk/oxdata/P01/EM/M2/J/*_Probabilities.h5 ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/J/

rsync -avz /Users/michielk/oxdata/P01/EM/M2/I/test_data_full_Probabilities.h5 ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/
rsync -avz /Users/michielk/oxdata/P01/EM/M2/I/training_data0_slicvoxels002.h5 ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/

rsync -avz /Users/michielk/workspace/EM_seg/src/EM_slicsegmentation_mpi.py ndcn0180@arcus.oerc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EMseg
rsync -avz /Users/michielk/oxscripts/P01/difsim/difsim_*.py ndcn0180@arcus.oerc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EMseg


rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/J/reg/ /Users/michielk/oxdata/P01/EM/M2/J/reg/
rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/J/*pdf /Users/michielk/oxdata/P01/EM/M2/J/
rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/*pdf /Users/michielk/oxdata/P01/EM/M2/I/

rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/test_data_full.h5 /Users/michielk/oxdata/P01/EM/M2/I/
rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/test_data_full_Probabilities.h5 /Users/michielk/oxdata/P01/EM/M2/I/

rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/test_data_slicsegmentation002_a100.stl /Users/michielk/oxdata/P01/EM/M2/I/

ssh -Y ndcn0180@arcus.oerc.ox.ac.uk
ssh -Y ndcn0180@caribou.oerc.ox.ac.uk

cd /home/ndcn-fmrib-water-brain/ndcn0180/workspace/EMseg
cd /data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I

# module load ImageJ/Dec2012
module load fiji/20140602
module load blender/2.72b__not-supported
module load python/2.7
module load hdf5/1.8.9
# module load hdf5-parallel/1.8.9_openmpi
module load matlab
# module load intel-compilers/2013
# module load intel-compilers/2012

module load python/2.7 mpi4py/1.3.1 hdf5-parallel/1.8.14_mvapich2


curl -O http://www.vtk.org/files/release/6.1/vtkpython-6.1.0-Linux-64bit.tar.gz
tar xvzf vtkpython-6.1.0-Linux-64bit.tar.gz
cd h5py; python setup.py install --user


# install cpg
rsync -avz /Users/michielk/Downloads/cgp13.zip ndcn0180@arcus.oerc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace

ssh -Y ndcn0180@arcus.oerc.ox.ac.uk
cd /home/ndcn-fmrib-water-brain/ndcn0180/workspace
module load python/2.7 mpi4py/1.3.1 hdf5-parallel/1.8.14_mvapich2
unzip cgp13.zip
mkdir cgp13-build
cd cgp13-build
module load cmake/2.8.12 matlab/R2014a vtk/5.10.1
ccmake ../cgp13
#manually set HDF5 library to: /system/software/arcus/lib/hdf5/1.8.14_mvapich2/lib/libhdf5.so.9








# local:
ImageJ --headless /Users/michielk/workspace/EM_seg/src/EM_convert2tif.py

python /Users/michielk/workspace/EMseg/EM_convert2stack.py \
-i '/Users/michielk/oxdata/P01/EM/M2/I/tifs' \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data.h5' \
-f 'stack' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448

ilastik --headless \
--project=/Users/michielk/oxdata/P01/EM/M2/I/training_data.ilp \
--output_internal_path=stack \
/Users/michielk/oxdata/P01/EM/M2/I/test_data_full.h5/stack

python /Users/michielk/workspace/EM_seg/src/EM_mpi_createh5.py \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_slicvoxels_c002_s1000_div50.h5' \
-g 'stack' \
-d 'uint32' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448
mpirun -n 1 python /Users/michielk/workspace/EM_seg/src/EM_slicvoxels_mpi.py \
-i '/Users/michielk/oxdata/P01/EM/M2/I/test_data.h5' \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_slicvoxels_c002_s1000_div50.h5' \
-f 'stack' \
-g 'stack' \
-s 1000 \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448

python /Users/michielk/workspace/EM_seg/src/EM_mpi_createh5.py \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_segmentation.h5' \
-g 'stack' \
-d 'uint32' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448
mpirun -n 8 python /Users/michielk/workspace/EM_seg/src/EM_gala_apply_classifier_mpi.py \
-p '/Users/michielk/oxdata/P01/EM/M2/I/test_data_Probabilities.h5' \
-s '/Users/michielk/oxdata/P01/EM/M2/I/test_data_slicvoxels.h5' \
-c '/Users/michielk/oxdata/P01/EM/M2/I/training_data_rf.pickle' \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_segmentation.h5' \
-f 'stack' \
-g 'stack' \
-x 400 -X 3600 -y 400 -Y 3600 -z 0 -Z 56

python /Users/michielk/workspace/EM_seg/src/EM_mpi_createh5.py \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_segmentation_relabel.h5' \
-g 'stack' \
-d 'uint16' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448

python /Users/michielk/workspace/EM_seg/src/EM_mpi_createh5.py \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_segmentation2.h5' \
-g 'stack' \
-d 'uint16' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448
mpirun -n 4 python /Users/michielk/workspace/EM_seg/src/EM_gala_apply_classifier_mpi.py \
-p '/Users/michielk/oxdata/P01/EM/M2/I/test_data_Probabilities.h5' \
-s '/Users/michielk/oxdata/P01/EM/M2/I/test_data_segmentation_relabel.h5' \
-c '/Users/michielk/oxdata/P01/EM/M2/I/training_data_rf.pickle' \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_segmentation2.h5' \
-f 'stack' \
-g 'stack' \
-x 0 -X 400 -y 0 -Y 400 -z 0 -Z 56


python /Users/michielk/workspace/EM_seg/src/EM_mpi_createh5.py \
-o '/Users/michielk/oxdata/P01/EM/M2/I/training_data0_slicsegmentation002.h5' \
-g 'stack' \
-d 'uint32' \
-x 0 -X 50 -y 0 -Y 500 -z 0 -Z 500
python /Users/michielk/workspace/EM_seg/src/EM_mpi_createh5.py \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_slicsegmentation002_a100.h5' \
-g 'stack' \
-d 'uint8' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448
python /Users/michielk/workspace/EM_seg/src/EM_slicsegmentation_seq.py \
-p '/Users/michielk/oxdata/P01/EM/M2/I/test_data_Probabilities.h5' \
-s '/Users/michielk/oxdata/P01/EM/M2/I/test_data_slicvoxels_c002_s1000_div50.h5' \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_slicsegmentation002.h5' \
-a 100 \
-f 'stack' \
-g 'stack' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448
mpirun -n 5 python /Users/michielk/workspace/EM_seg/src/EM_slicsegmentation_mpi.py \
-p '/Users/michielk/oxdata/P01/EM/M2/I/test_data_Probabilities.h5' \
-s '/Users/michielk/oxdata/P01/EM/M2/I/test_data_slicvoxels_c002_s1000_div50.h5' \
-o '/Users/michielk/oxdata/P01/EM/M2/I/test_data_slicsegmentation002_a100.h5' \
-a 100 \
-f 'stack' \
-g 'stack' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 448


blender -b -P /Users/michielk/workspace/EM_seg/src/EM_blender.py
/Users/michielk/workspace/DifSim/mcell/bin/mcell /Users/michielk/oxdata/P01/EM/M2/I/test_data_slicsegmentation002_a100_files/mcell/Scene.main.mdl



datadir = '/Users/michielk/oxdata/P01/EM/M2/I/'
fp = h5py.File(os.path.join(datadir, 'training_data0_ws_Probabilities.h5'),'r+')
fp['volume/predictions'] = h5py.SoftLink('/stack')
fp.close()
gala-segmentation-pipeline --segmentation-thresholds 0.5 \
--image-stack '/Users/michielk/oxdata/P01/EM/M2/I/training_data0_ws_SimpleSegmentation.h5' \
--disable-gen-pixel --pixelprob-file '/Users/michielk/oxdata/P01/EM/M2/I/training_data0_ws_Probabilities.h5' \
--enable-gen-supervoxels \
--enable-gen-agglomeration \
--disable-inclusion-removal \
--disable-raveler-output \
--enable-h5-output '/Users/michielk/oxdata/P01/EM/M2/I/training_data0_ws_segmentation'


gala-segmentation-pipeline --segmentation-thresholds 0.5 \
--image-stack '/Users/michielk/oxdata/P01/EM/M2/I/training_data0_ws_SimpleSegmentation.h5' \
--disable-gen-pixel --pixelprob-file '/Users/michielk/oxdata/P01/EM/M2/I/training_data0_ws_Probabilities.h5' \
--disable-gen-supervoxels --supervoxels-name='/Users/michielk/oxdata/P01/EM/M2/I/training_data0_slicvoxels.h5' \
--enable-gen-agglomeration \
--disable-inclusion-removal \
--disable-raveler-output \
--enable-h5-output '/Users/michielk/oxdata/P01/EM/M2/I/training_data0_ws_segmentation'
















### convert to tifs and rename to simple filename base
from os import path
import glob
from loci.plugins import BF
from ij import IJ

# Get list of DM3 files
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/J'
indir = 'orig'
outdir = 'tifs'
infiles = glob.glob(path.join(datadir, indir, '*.dm3'))

for infile in infiles:
    imp = BF.openImagePlus(infile)
    IJ.save(imp[0], path.join(datadir, outdir, infile[-8:-4] + '.tif'))




### register the slices in the stack to one another (working from X11)
from register_virtual_stack import Register_Virtual_Stack_MT

source_dir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/J/tifs/"
target_dir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/J/reg/"
transf_dir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/J/reg/trans/"
reference_name = '0001.tif'
use_shrinking_constraint = 0

p = Register_Virtual_Stack_MT.Param()
p.sift.maxOctaveSize = 1024
p.minInlierRatio = 0.05

Register_Virtual_Stack_MT.exec(source_dir, target_dir, transf_dir, reference_name, p, use_shrinking_constraint)

