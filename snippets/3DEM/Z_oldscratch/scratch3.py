scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/M3/M3_S1_GNU/testblock"
dataset='m000'

(430, 4460, 5217)
rename 00460 00430 \
${dataset}_?????-?????_?????-?????_?????-?????*.*
rename ${dataset}_5000-6000 ${dataset}_5000-5217 \
${dataset}_?????-?????_?????-?????_?????-?????*.*
rename _04000-05000_00000-00430 _04000-04460_00000-00430 \
${dataset}_?????-?????_?????-?????_?????-?????*.*

#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_04000-05000_04000-05000_00000-00460*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
z=0; Z=430;
for x in 0 1000 2000 3000 4000; do
[ $x == 5000 ] && X=5217 || X=$((y+1000))
qsubfile=$datadir/EM_p2l_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > $datadir/output_${x}-${X}_${y}-${Y} &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile
done
#m000_04000-05000_04000-05000_00000-00460_probs0_eed2.h5



### 2x2 sections processing

for file in `ls $datadir/${dataset}_*-*_*-*_0-100.h5`; do
echo $file
cp $file ${file/.h5/_.h5}
done
pf=
qsubfile=$datadir/EM_mb_${pf}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_0-1000_0-1000_*-*_${pf}.h5 \
$datadir/${dataset}_0-1000_1000-2000_*-*_${pf}.h5 \
$datadir/${dataset}_1000-2000_0-1000_*-*_${pf}.h5 \
$datadir/${dataset}_1000-2000_1000-2000_*-*_${pf}.h5 \
-o $datadir/${dataset}_0x0.h5 \
-f 'stack' -l 'zyx'" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_0-1000_2000-3000_*-*_${pf}.h5 \
$datadir/${dataset}_0-1000_3000-4000_*-*_${pf}.h5 \
$datadir/${dataset}_1000-2000_2000-3000_*-*_${pf}.h5 \
$datadir/${dataset}_1000-2000_3000-4000_*-*_${pf}.h5 \
-o $datadir/${dataset}_0x2.h5 \
-f 'stack' -l 'zyx'" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_2000-3000_0-1000_*-*_${pf}.h5 \
$datadir/${dataset}_2000-3000_1000-2000_*-*_${pf}.h5 \
$datadir/${dataset}_3000-4000_0-1000_*-*_${pf}.h5 \
$datadir/${dataset}_3000-4000_1000-2000_*-*_${pf}.h5 \
-o $datadir/${dataset}_2x0.h5 \
-f 'stack' -l 'zyx'" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_2000-3000_2000-3000_*-*_${pf}.h5 \
$datadir/${dataset}_2000-3000_3000-4000_*-*_${pf}.h5 \
$datadir/${dataset}_3000-4000_2000-3000_*-*_${pf}.h5 \
$datadir/${dataset}_3000-4000_3000-4000_*-*_${pf}.h5 \
-o $datadir/${dataset}_2x2.h5 \
-f 'stack' -l 'zyx'" >> $qsubfile
sbatch -p devel $qsubfile

rm *_.h5
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/archive/m000_z0000-z0100/m000_?x?.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/

for pf in probs_ws_MM probs_ws_MA probs_ws_UA probs_ws_PA; do
qsubfile=$datadir/EM_mb_${pf}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_0-1000_0-1000_*-*_${pf}.h5 \
$datadir/${dataset}_0-1000_1000-2000_*-*_${pf}.h5 \
$datadir/${dataset}_1000-2000_0-1000_*-*_${pf}.h5 \
$datadir/${dataset}_1000-2000_1000-2000_*-*_${pf}.h5 \
-o $datadir/${dataset}_0x0_${pf}.h5 \
-f 'stack' -l 'zyx'" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_0-1000_2000-3000_*-*_${pf}.h5 \
$datadir/${dataset}_0-1000_3000-4000_*-*_${pf}.h5 \
$datadir/${dataset}_1000-2000_2000-3000_*-*_${pf}.h5 \
$datadir/${dataset}_1000-2000_3000-4000_*-*_${pf}.h5 \
-o $datadir/${dataset}_0x2_${pf}.h5 \
-f 'stack' -l 'zyx'" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_2000-3000_0-1000_*-*_${pf}.h5 \
$datadir/${dataset}_2000-3000_1000-2000_*-*_${pf}.h5 \
$datadir/${dataset}_3000-4000_0-1000_*-*_${pf}.h5 \
$datadir/${dataset}_3000-4000_1000-2000_*-*_${pf}.h5 \
-o $datadir/${dataset}_2x0_${pf}.h5 \
-f 'stack' -l 'zyx'" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_2000-3000_2000-3000_*-*_${pf}.h5 \
$datadir/${dataset}_2000-3000_3000-4000_*-*_${pf}.h5 \
$datadir/${dataset}_3000-4000_2000-3000_*-*_${pf}.h5 \
$datadir/${dataset}_3000-4000_3000-4000_*-*_${pf}.h5 \
-o $datadir/${dataset}_2x2_${pf}.h5 \
-f 'stack' -l 'zyx'" >> $qsubfile
sbatch -p devel $qsubfile
done
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/archive/m000_z0000-z0100/m000_?x?_*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
for dataset in m000_0x0 m000_0x2 m000_2x0 m000_2x2; do
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.nii.gz" -i 'zyx' -l 'xyz'
for pf in probs_ws_MM probs_ws_MA probs_ws_UA probs_ws_PA; do
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_${pf}.h5" \
"${datadir}/${dataset}_${pf}.nii.gz" -i 'zyx' -l 'xyz'
done
done


dataset='m000'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.nii.gz" -f "/stack" -i 'zyx' -l 'xyz'



qsubfile=$datadir/EM_ac2s2.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=100:00:00" >> $qsubfile
echo "#SBATCH --mem=250000" >> $qsubfile
echo "#SBATCH --job-name=EM_il" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate ilastik-devel" >> $qsubfile
echo "CONDA_ROOT=\`conda info --root\`" >> $qsubfile
echo "\${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/${pixprob_trainingset}_arcus.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/${dataset}_probs.h5 \
--output_internal_path=/volume/predictions \
$datadir/${dataset}.h5/stack" >> $qsubfile
sbatch -p compute $qsubfile



rename 0.h5 00460.h5 m000_*-*_*-*_*-0.h5

m000_01000-02000_01000-02000_00000-00460.h5



(100, 4111, 4235)
rename 4000-5000 4000-4111 m000_*_4000-5000_0-100*.*
rename 4000-5000 4000-4235 m000_4000-5000_*_0-100*.h5
rename m000_4000-4111 m000_4000-4235 m000_4000-4111_4000-4111_0-100*
rename m000_4000-5000 m000_4000-4235 m000_4000-5000_*_0-100_probs.log

scriptdir="$HOME/workspace/EM"
oddir=$DATA/EM/M3/20Mar15/montage/Montage_
datadir="$DATA/EM/M3/M3_S1_GNU/archive/m000_z0000-z0100"
dataset='m000'
refsect='0250'
pixprob_trainingset="pixprob_training"

cp $datadir/${dataset}_0-1000_0-1000_0-100.h5 $datadir/${dataset}_0-1000_0-1000_0-100_.h5
cp $datadir/${dataset}_0-1000_1000-2000_0-100.h5 $datadir/${dataset}_0-1000_1000-2000_0-100_.h5
cp $datadir/${dataset}_1000-2000_0-1000_0-100.h5 $datadir/${dataset}_1000-2000_0-1000_0-100_.h5
cp $datadir/${dataset}_1000-2000_1000-2000_0-100.h5 $datadir/${dataset}_1000-2000_1000-2000_0-100_.h5
qsubfile=$datadir/EM_mb.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_0-1000_0-1000_*-*_.h5 \
$datadir/${dataset}_0-1000_1000-2000_*-*_.h5 \
$datadir/${dataset}_1000-2000_0-1000_*-*_.h5 \
$datadir/${dataset}_1000-2000_1000-2000_*-*_.h5 \
-o $datadir/${dataset}_2x2.h5 \
-f 'stack' -e 0.05 0.0073 0.0073 -l 'zyx'" >> $qsubfile
sbatch -p devel $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/archive/m000_z0000-z0100/m000_2x2.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/



qsubfile=$datadir/EM_mb.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_*-*_*-*_*-*_probs.h5 \
-f 'volume/predictions' -e 0.05 0.0073 0.0073 1 -l 'zyxc'" >> $qsubfile
sbatch -p devel $qsubfile


qsubfile=$datadir/EM_mb.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_0-1000_0-1000_*-*_probs.h5 \
$datadir/${dataset}_0-1000_1000-2000_*-*_probs.h5 \
$datadir/${dataset}_1000-2000_0-1000_*-*_probs.h5 \
$datadir/${dataset}_1000-2000_1000-2000_*-*_probs.h5 \
-o $datadir/${dataset}_probs_2x2.h5 \
-f 'volume/predictions' -e 0.05 0.0073 0.0073 1 -l 'zyxc'" >> $qsubfile
sbatch -p devel $qsubfile

qsubfile=$datadir/EM_s2s.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}_probs_2x2.h5 \
$datadir/${dataset}_probs_2x2_prob0.h5 \
-f 'volume/predictions' -g 'volume/predictions' -s 20 20 20 1 -i zyxc -l zyxc \
-c 0 -C 1" >> $qsubfile
sbatch -p devel $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/archive/m000_z0000-z0100/m000_probs_2x2_prob0.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/


for layer in 0 1 2; do
qsubfile=$datadir/EM_mb.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_*-*_*-*_*-*_probs${layer}_eed2.h5 \
-f 'stack' -e 0.05 0.0073 0.0073 -l 'zyx'" >> $qsubfile
sbatch -p devel $qsubfile
done
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/archive/m000_z0000-z0100/m000_eed2.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/



python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_*-*_*-*_*-*_probs.h5 \
-f 'volume/predictions' -e 0.05 0.0073 0.0073 1 -l 'zyxc'



# m000_1000-2000_1000-2000_0-100_probs0_eed2.h5
# m000_3000-4000_2000-3000_0-100_probs1_eed2.h5
# m000_4000-5000_2000-3000_0-100_probs0_eed2.h5

./dojo.py /PATH/TO/MOJO/FOLDER/WITH/IDS/AND/IMAGES/SUBFOLDERS



qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=s2s" >> $qsubfile
echo "python $scriptdir/convert/EM_series2stack.py \
$datadir/${dataset}_reg $datadir/${dataset}_reg_testnonmpi.h5 \
-f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o" >> $qsubfile
sbatch -p devel $qsubfile



z=0; Z=100; x=0; X=1000; y=0; Y=1000; layer=1;
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/snippets/eed/EM_eed.m
qsubfile=$datadir/EM_eed_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
echo "$datadir/EM_eed '$datadir' '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs' '/volume/predictions' 'stack' 1 &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile



./EM_eed $datadir ${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs '/volume/predictions' 'stack' 1

datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU';
invol = 'm000_0-1000_0-1000_0-100_probs';
invol = 'm000_0-1000_0-1000_0-100';
infield = '/volume/predictions';
infield = '/stack';
outfield = '/stack';
layers = [1,2,3];
layers = 0;
addpath(genpath('~/oxscripts/matlab/toolboxes/coherencefilter_version5b'));




$datadir/run_EM_eed.sh "$datadir" 'm000_0-1000_0-1000_0-100_probs' '/volume/predictions' 'stack' 1




python $scriptdir/convert/EM_mergeblocks.py

x=500; X=1500; y=0; Y=1000; z=0; Z=100;
python $scriptdir/convert/EM_stack2stack.py \
$datadir/m000_probs.h5 $datadir/m000_${x}-${X}_${y}-${Y}_${z}-${Z}_probs.h5 \
-f '/volume/predictions' -g '/volume/predictions' -s 20 20 20 6 -i zyxc -l zyxc -e 0.05 0.0073 0.0073 1 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z





mpiexec -n 4 python $scriptdir/convert/EM_series2stack.py $datadir/${dataset}_reg $datadir/${dataset}_reg.h5 -f 'stack' -m -o -e 0.0073 0.0073 0.05 -c 20 20 40


# python $scriptdir/convert/EM_series2stack.py $datadir/${dataset}_reg $datadir/${dataset}_reg.h5 -f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o
# mpiexec -n 4 python $scriptdir/convert/EM_series2stack.py $datadir/${dataset}_reg $datadir/${dataset}_reg.h5 -f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o -m

qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=12" >> $qsubfile
echo "#PBS -l walltime=12:00:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_series2stack.py \
$datadir/${dataset}_reg $datadir/${dataset}_reg.h5 \
-f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o -m" >> $qsubfile
qsub $qsubfile



###==========================================###
### downsample registered slices for viewing ###
###==========================================###
# NOTE: ONLY WORKING ON ARCUS
module load python/2.7
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2

scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU

mkdir -p $datadir/m000_reg_ds

qsubfile=$datadir/EM_downsample_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_reg_ds" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_downsample.py \
'$datadir/m000_reg' '$datadir/m000_reg_ds' -d 10 -m" >> $qsubfile
qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_reg_ds/* /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_reg_ds_arcus/

qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=16" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_series2stack.py \
'$datadir/m000_reg_ds' '$datadir/m000_reg_ds.h5' \
-f 'stack' -m -o -e 0.073 0.073 0.05" >> $qsubfile
qsub -q develq $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_reg_ds.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_reg_ds_arcus.h5



z=0; Z=100;
for x in 0 1000 2000 3000 4000; do
X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
qsubfile=$datadir/EM_eed_${x}-${X}_${y}-${Y}_${z}-${Z}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
echo "matlab -nojvm -singleCompThread -r \"addpath('$scriptdir/snippets/eed'); \
EM_eed('$datadir', '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs', '/volume/predictions', 'stack', 1); \
EM_eed('$datadir', '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs', '/volume/predictions', 'stack', 2); \
EM_eed('$datadir', '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs', '/volume/predictions', 'stack', 3); \
exit\""  >> $qsubfile
sbatch -p compute $qsubfile
done
done
