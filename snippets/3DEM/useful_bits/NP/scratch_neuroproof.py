#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/pixprob_training.ilp ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/pixprob_training.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/
### run ilastik to adapt .ilp file ###
export PATH=$DATA/miniconda/bin:$PATH
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh


###======================###
### gala for supervoxels ###
###======================###
z=30; Z=460;
qsubfile=$datadir/EM_gala.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_gala" >> $qsubfile
for x in 1000 1500; do
for y in 1000 1500; do
[ $x == 5000 ] && X=5217 || X=$((x+500))
[ $y == 4000 ] && Y=4460 || Y=$((y+500))
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
echo "/home/ndcn-fmrib-water-brain/ndcn0180/.local/bin/gala-segmentation-pipeline \
--image-stack ${datadir}/${datastem}.h5 \
--ilp-file ${datadir}/${pixprob_trainingset}_arcus.ilp \
--disable-gen-pixel --pixelprob-file ${datadir}/${datastem}_probs.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 5 \
--enable-raveler-output \
--enable-h5-output ${datadir}/${datastem} \
--segmentation-thresholds 0.0" >> $qsubfile
done
done
sbatch -p devel $qsubfile

#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_0??00-0??00_0??00-0??00_00030-00460_slic_s00500_c0.020_o0.010.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/
###================================###
### Neuroproof classifier training ###
###================================###
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;  # 22GB training for pfsvox='_slic_s00500_c2.000_o0.050'
x=1000; X=1500; y=1000; Y=1500; z=200; Z=300;  # ??GB training for pfsvox='_slic_s00500_c2.000_o0.050'
x=1000; X=1100; y=1000; Y=1100; z=200; Z=300;  # ??GB training for pfsvox='_slic_s00500_c2.000_o0.050'
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
pfsvox='_slic_s00500_c2.000_o0.050'  # pfsvox='/supervoxels'
strat=3; niter=2
qsubfile=$datadir/EM_nplearn_strat${strat}_n${niter}_${datastem}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
# echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_nplearn" >> $qsubfile
echo "export PATH=$DATA/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate neuroproof-devel" >> $qsubfile
echo "neuroproof_graph_learn \
${datadir}/${datastem}${pfsvox}.h5 \
${datadir}/${datastem}_probs_xyzc.h5 \
${datadir}/${datastem}_PA.h5 \
--classifier-name ${datadir}/${datastem}${pfsvox}_classifier_str${strat}_n${niter}.xml \
--strategy-type ${strat} --num-iterations ${niter}" >> $qsubfile
sbatch -p devel $qsubfile
sbatch -p compute $qsubfile

#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_classifier_str*  /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1100; y=1000; Y=1100; z=200; Z=300;  # ??GB training for pfsvox='_slic_s00500_c2.000_o0.050'
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"

gala-segmentation-pipeline \
--image-stack ${datadir}/${datastem}.h5 \
--ilp-file ${datadir}/${pixprob_trainingset}.ilp \
--disable-gen-pixel --pixelprob-file ${datadir}/${datastem}_probs.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 5 \
--enable-raveler-output \
--enable-h5-output ${datadir}/${datastem} \
--segmentation-thresholds 0.0

pfsvox='_slic_s00500_c2.000_o0.050'  #
pfsvox='/supervoxels'
source activate neuroproof
cd /Users/michielk/anaconda/envs/neuroproof/  # neuroproof only seems to work when in the neuroproof directory!!
source activate neuroproof-devel
cd /Users/michielk/anaconda/envs/neuroproof-devel/  # neuroproof only seems to work when in the neuroproof directory!!
strat=2; niter=2
neuroproof_graph_learn \
${datadir}/${datastem}${pfsvox}.h5 \
${datadir}/${datastem}_probs_xyzc.h5 \
${datadir}/${datastem}_PA.h5 \
--classifier-name ${datadir}/${datastem}${pfsvox}_classifier_str${strat}_n${niter}.xml \
--strategy-type ${strat} --num-iterations ${niter}

###=======================###
### Neuroproof prediction ###
###=======================###
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
x=1000; X=1500; y=1500; Y=2000; z=30; Z=460;
x=1500; X=2000; y=1000; Y=1500; z=30; Z=460;  # 38GB prediction for pfsvox='_slic_s00500_c2.000_o0.050' in first 10 min
x=1500; X=2000; y=1500; Y=2000; z=30; Z=460;
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
classifier="m000_01000-01500_01000-01500_00030-00460_slic_s00500_c2.000_o0.050_classifier_str1.xml"
pfsvox='_slic_s00500_c2.000_o0.050'  # pfsvox='/supervoxels'
qsubfile="${datadir}/EM_np-pred_${datastem}.sh"
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=10:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_np-pred_${datastem}" >> $qsubfile
echo "export PATH=${HOME}/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate neuroproof-devel" >> $qsubfile
echo "neuroproof_graph_predict \
${datadir}/${datastem}${pfsvox}.h5 \
${datadir}/${datastem}_probs_xyzc.h5 \
${datadir}/${classifier} \
--output-file ${datadir}/${datastem}_segmentation.h5 \
--graph-file ${datadir}/${datastem}_graph.json" >> $qsubfile
sbatch -p compute $qsubfile



import numpy as np
import h5py
from skimage.segmentation import relabel_sequential
from skimage.morphology import remove_small_objects

datadir = "/users/michielk/oxdata/P01/EM/M3/M3_S1_GNU"
datastem = "m000_01000-01500_01000-01500_00030-00460"
datastem = "m000_01000-01500_01500-02000_00030-00460"
label_field = loadh5(datadir, datastem + "_segmentation")[0]
len(np.unique(label_field))
remove_small_objects(label_field, min_size=50000, connectivity=1, in_place=True)
relabeled = relabel_sequential(label_field)[0]
len(np.unique(label_field))
writeh5(relabeled, datadir, datastem + "_segmentation_rl", element_size_um=[0.05,0.0073,0.0073])
