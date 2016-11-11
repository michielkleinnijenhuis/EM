datadir='/Users/michielk/M3_S1_GNU_NP/train'
datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP/train'
dset_name='m000_01000-01500_01000-01500_00030-00460'
datadir='/Users/michielk/M3_S1_GNU_NP/test'
datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP/test'
dset_name='m000_02000-03000_02000-03000_00030-00460'
dset_name='m000_03000-04000_03000-04000_00030-00460'

### preamble
data, elsize = loadh5(datadir, dset_name, fieldname='/stack')
datamask = data != 0
datamask = binary_dilation(binary_fill_holes(datamask))  # TODO
writeh5(datamask, datadir, dset_name + '_maskDS', dtype='uint8', element_size_um=elsize)

probmask = prob[:,:,:,0] > 0.2
prob, elsize = loadh5(datadir, dset_name + '_probs', fieldname='/volume/predictions')
probmask = prob[:,:,:,0] > 0.2
writeh5(probmask, datadir, dset_name + '_maskMM', dtype='uint8', element_size_um=elsize)
probmask = prob[:,:,:,3] > 0.3
writeh5(probmask, datadir, dset_name + '_maskMB', dtype='uint8', element_size_um=elsize)

### postproc
per, elsize = loadh5(datadir, dset_name + '_per')
input_watershed='_ws_l0.95_u1.00_s064'
ws = loadh5(datadir, dset_name + input_watershed)[0]
input_watershed='_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1'
pred = loadh5(datadir, dset_name + input_watershed)[0]
if 0:
    input_MA='_maskMA'
    MAmask = loadh5(datadir, dset_name + input_MA)[0]
else:
    MAmask = np.zeros_like(per, dtype='uint8')
    MAmask[per>0.5] = 1
    writeh5(MAmask, datadir, dset_name + '_classMA')

#pred m000_03000-04000_03000-04000_00030-00460
MMlabs = [214, 197, 180, 117, 84, 79, 72, 81, 70, 9, 102, 116, 319]
MMlabs = [161, 992, 990, 235, 811, 290, 1314, 1850, 22, 1658, 2741, 3477]
MMlabs = [3050, 2962, 2267, 3102, 4191, 2429, 3839, 3875, 3865, 4347, 3555, 4369, 4441]
MMlabs = [2171, 3138, 3986, 3752, 3125, 3806, 3792, 3914, 3721, 3484, 3602, 2651, 2578, 1295, 1889]
MMlabs = [232, 454, 593, 735, 561, 2266, 1923, 2180, 2179, 2003, 2621, 3620, 2771]
MMlabs = [1412, 1666, 1227, 2355, 2488]
for l in MMlabs:
    MAmask[pred==l] = 1

UAlabs = [117, 22]
UAlabs = [1904, 161]
for l in UAlabs:
    MAmask[pred==l] = 0

# uncertain 72, 102, 1850, 2396

#ws m000_03000-04000_03000-04000_00030-00460
MMlabs = [161, 583, 615, 544, 437, 1242, 2158, 1934, 1700, 2476, 2488, 2498, 2355]
MMlabs = [1508, 669, 764, 989, 1234, 1003, 1681, 2141, 2289, 2275, 2357, 3451, 2610, 3263, 3381, 3654]
MMlabs = [163, 1225, 2384, 2372, 2346, 2820, 2736, 2522, 2765, 3410, 3700, 3961, 4038, 4447]
for l in MMlabs:
    MAmask[ws==l] = 1

UAlabs = [1003, 1850, 4191]
for l in UAlabs:
    MAmask[ws==l] = 0

# uncertain 2139

writeh5(MAmask, datadir, dset_name + '_MA')


#pred m000_02000-03000_02000-03000_00030-00460
MMlabs = [165, 498, 245, 447, 349, 2064, 1688, 636, 781, 941, 721, 1266, 1424, 811, 2270, 1947, 2638, 2064, 2895, 2612, 2878, 2612, 2375, 3017]
MMlabs = [1704, 1621, 1871, 2863, 3167, 3470, 3315]
for l in MMlabs:
    MAmask[pred==l] = 1

UAlabs = [165, 811]
for l in UAlabs:
    MAmask[pred==l] = 0

# uncertain 219, 25

# TODO
outpf = outpf + '_filled'
MA = fill_holes(MAmask)
writeh5(MAmask, datadir, dset_name + outpf, element_size_um=elsize)
MMmask[MAmask != 0] = False


datadir='/Users/michielk/M3_S1_GNU_NP/test'


scriptdir="$HOME/workspace/EM"
datadir='/Users/michielk/M3_S1_GNU_NP/test'
dataset='m000'
python $scriptdir/mesh/EM_separate_sheaths.py \
$datadir $dataset \
--maskDS '_maskDS' 'stack' \
--maskMM '_maskMM' 'stack' \
--maskMA '_maskMA' 'stack' \
--supervoxels '_supervoxels' 'stack' \
-x 2000 -X 3000 -y 2000 -Y 3000 -z 30 -Z 460





### local
scriptdir="$HOME/workspace/EM"
datadir='/Users/michielk/M3_S1_GNU_NP/train'
datastem='m000_01000-01500_01000-01500_00030-00460'
datadir='/Users/michielk/M3_S1_GNU_NP/test'
datastem='m000_02000-03000_02000-03000_00030-00460'
datastem='m000_03000-04000_03000-04000_00030-00460'
# mpiexec -n 6 python $scriptdir/mesh/EM_classify_neurons.py \
# $datadir $datastem -p '_probs' -f '/volume/predictions' -c 0 3 -w '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1' -o '_per' -l 0.2 0.3 -m
mpiexec -n 6 python $scriptdir/mesh/EM_classify_neurons.py \
$datadir $datastem -w '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1' -o '_per' -m

### ARC
# ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
source ~/.bashrc
module load python/2.7__gcc-4.8
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2_gcc

scriptdir="$HOME/workspace/EM"
datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP'
cd $datadir
ddir="$datadir/train"
datastem=m000_01000-01500_01000-01500_00030-00460
ddir="$datadir/test"
datastem=m000_02000-03000_02000-03000_00030-00460
ddir="$datadir/test"
datastem=m000_03000-04000_03000-04000_00030-00460

q=d
qsubfile=$datadir/classify.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=8" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00" || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=classify" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/mesh/EM_classify_neurons.py \
$ddir $datastem \
--supervoxels '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1' 'stack' \
-o '_per' -m" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile


# # ssh -Y ndcn0180@arcus.arc.ox.ac.uk
# source ~/.bashrc
# module load python/2.7__gcc-4.8
# module load mpi4py/1.3.1
# module load hdf5-parallel/1.8.14_mvapich2
# q=d
# qsubfile=$datadir/classify.sh
# echo '#!/bin/bash' > $qsubfile
# echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
# [ "$q" = "d" ] && echo "#PBS -l walltime=00:10:00" || echo "#PBS -l walltime=10:00:00" >> $qsubfile
# echo "#PBS -N classify" >> $qsubfile
# echo "#PBS -V" >> $qsubfile
# echo "cd \$PBS_O_WORKDIR" >> $qsubfile
# echo ". enable_arcus_mpi.sh" >> $qsubfile
# echo "echo `which python`" >> $qsubfile
# # echo "mpirun \$MPI_HOSTS python $scriptdir/mesh/EM_classify_neurons.py \
# # $ddir $datastem -p '_probs' -f '/volume/predictions' -c 0 3 \
# # -w '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1' \
# # -o '_per' -l 0.2 0.3 -m" >> $qsubfile
# [ "$q" = "d" ] && qsub -q develq $qsubfile || qsub $qsubfile







import os
import sys
from argparse import ArgumentParser
import h5py
import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_fill_holes
from skimage.morphology import dilation
from skimage.measure import regionprops
from scipy.ndimage.filters import gaussian_filter

def loadh5(datadir, dname, fieldname='stack'):
    """"""
    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')
    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:,:]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:,:,:]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:,:,:,:]
    if 'element_size_um' in f[fieldname].attrs.keys():
        element_size_um = f[fieldname].attrs['element_size_um']
    else:
        element_size_um = None
    f.close()
    return stack, element_size_um


def writeh5(stack, datadir, fp_out, fieldname='stack', dtype='uint16', element_size_um=None):
    """"""
    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")
    if len(stack.shape) == 2:
        g[fieldname][:,:] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:,:,:] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:,:,:,:] = stack
    if element_size_um is not None:
        g[fieldname].attrs['element_size_um'] = element_size_um
    g.close()


def fill_holes(MA):
    """"""
    for l in np.unique(MA)[1:]:
        # fill holes
        labels = label(MA!=l)[0]
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        MA[labels != background] = l
        # closing
        binim = MA==l
        binim = binary_closing(binim, iterations=10)
        MA[binim] = l
        # fill holes
        labels = label(MA!=l)[0]
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        MA[labels != background] = l

    return MA



datadir='/Users/michielk/M3_S1_GNU_NP/train'
dset_name='m000_01000-01500_01000-01500_00030-00460'
# input_prob='_probs0_eed2'
# fieldnamein='stack'
# lower_threshold=0.2
# prob, elsize = loadh5(datadir, dset_name + input_prob, fieldname=fieldnamein)
# probmask = prob > lower_threshold
myelmask, elsize = loadh5(datadir, dset_name + '_myelin')
memmask, elsize = loadh5(datadir, dset_name + '_membrane')

input_watershed='_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1'
ws = loadh5(datadir, dset_name + input_watershed)[0]

rp = regionprops(ws)
areas = [prop.area for prop in rp]

wsdil = dilation(ws)
rpdil = regionprops(wsdil)
areasdil = [prop.area for prop in rpdil]

# np.greater(areas, areasdil)  # probably not myelin

wsdilmask = np.copy(wsdil)
wsdilmask[probmask==False] = 0
areasdilmask = [np.sum(wsdilmask==l) for l in labels]

# labels = np.unique(ws)[1:]
# labelsdilmask = np.unique(wsdilmask)[1:]
# lostlabels = set(labels) - set(labelsdilmask)  # not touching mask
# rpdilmask = regionprops(wsdilmask)
# areasdilmask = [prop.area if prop.label in labelsdilmask else 0 for prop in rpdilmask]

per = np.divide(areasdilmask, areas, dtype='float')

areasdiff = np.subtract(areasdil, areas)
per = np.divide(areasdilmask, areasdiff, dtype='float')

perc = np.zeros_like(ws, dtype='float')
for i, l in enumerate(labels):
    perc[ws==l] = per[i]

output_postfix='_perc2'
writeh5(perc, datadir, dset_name + output_postfix, dtype='float', element_size_um=elsize)

per[np.where(labels==62)]





## axon classification in MA and UA
# dilate every label
# compute the number of voxels in the dilation
# evaluate the percentage of voxels overlapping myelin

scriptdir="$HOME/workspace/EM"
datadir='/Users/michielk/M3_S1_GNU_NP/train/orig'
datastem='m000_01000-01500_01000-01500_00030-00460'
python $scriptdir/mesh/EM_classify_neurons.py $datadir $datastem

# mpiexec -n 8 python $scriptdir/mesh/EM_classify_neurons.py $datadir $dset_name -m

mpiexec -n 8 python $scriptdir/mesh/EM_classify_neurons.py \
$datadir $datastem -p '_probs0_eed2' -w '_ws' -o '_per0' -l 0.2 -m

mpiexec -n 8 python $scriptdir/mesh/EM_classify_neurons.py \
$datadir $datastem -p '_probs' -f '/volume/predictions' -c 3 -w '_ws' -o '_per3' -l 0.3 -m

scriptdir="$HOME/workspace/EM"
datadir='/Users/michielk/M3_S1_GNU_NP/train'
datastem='m000_01000-01500_01000-01500_00030-00460'
# mpiexec -n 8 python $scriptdir/mesh/EM_classify_neurons.py \
# $datadir $datastem -p '_probs0_eed2' -w '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1' -o '_per0' -l 0.2 -m
# mpiexec -n 8 python $scriptdir/mesh/EM_classify_neurons.py \
# $datadir $datastem -p '_probs' -f '/volume/predictions' -c 3 -w '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1' -o '_per3' -l 0.3 -m
mpiexec -n 8 python $scriptdir/mesh/EM_classify_neurons.py \
$datadir $datastem -p '_probs' -f '/volume/predictions' -c 0 3 -w '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1' -o '_per' -l 0.2 0.3 -m



# combine percentages
datadir='/Users/michielk/M3_S1_GNU_NP/train/orig'
dset_name='m000_01000-01500_01000-01500_00030-00460'

data, elsize = loadh5(datadir, dset_name)
per0 = loadh5(datadir, dset_name + '_per0')[0]
per3 = loadh5(datadir, dset_name + '_per3')[0]
ws = loadh5(datadir, dset_name + '_ws')[0]
MA = np.zeros_like(ws)

for l in np.unique(ws)[1:]:
    print(l)
    p0 = per0[ws==l][0]
    p3 = per3[ws==l][0]
    if (p0 > 0.5 and p3 < 0.5):
        MA[ws==l] = l

writeh5(MA, datadir, dset_name + '_MAonly', element_size_um=elsize)

pf="_MAonly"
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.05 0.0073 0.0073 -i 'zyx' -l 'zyx'
