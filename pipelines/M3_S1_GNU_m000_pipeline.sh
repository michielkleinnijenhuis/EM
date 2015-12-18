### arcus-b ###

###=====================###
### prepare environment ###
###=====================###


# ssh -Y ndcn0180@arcus.arc.ox.ac.uk
source ~/.bashrc
module load python/2.7
module load mpi4py/1.3.1 
module load hdf5-parallel/1.8.14_mvapich2

# ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
source ~/.bashrc
module load hdf5-parallel/1.8.14_mvapich2
module load mvapich2/2.0.1__intel-2015
module load mpi4py/1.3.1
module load python/2.7__gcc-4.8
module load matlab/R2015a
# sacct --format=JobID,Timelimit,elapsed,ReqMem,MaxRss,AllocCPUs,AveVMSize,MaxVMSize,CPUTime,ExitCode | less


###=====================###
### variable definition ###
###=====================###

scriptdir="$HOME/workspace/EM"
oddir=$DATA/EM/M3/20Mar15/montage/Montage_
datadir="$DATA/EM/M3/M3_S1_GNU"
dataset='m000'
refsect='0250'
pixprob_trainingset="pixprob_training"


###=====================###
### convert DM3 to tifs ###
###=====================###

mkdir -p $datadir/tifs

for montage in 000 001 002 003; do
sed "s?INPUTDIR?$oddir$montage?;\
    s?OUTPUTDIR?$datadir/tifs?;\
    s?OUTPUT_POSTFIX?_m$montage?g" \
    $scriptdir/EM_tiles2tif.py \
    > $datadir/EM_tiles2tif_m$montage.py
done

sed "s?DATADIR?$datadir?g" \
    $scriptdir/EM_tiles2tif_submit.sh \
    > $datadir/EM_tiles2tif_submit.sh

qsub -t 0-3 $datadir/EM_tiles2tif_submit.sh

mkdir -p $datadir/${dataset}
cp $datadir/${dataset}/tifs/*_${dataset}.tif $datadir/tifs_${dataset}/


###=================###
### register slices ###
###=================###
mkdir -p $datadir/${dataset}_reg/trans

sed "s?SOURCE_DIR?$datadir/tifs_${dataset}?;\
    s?TARGET_DIR?$datadir/tifs_${dataset}_reg?;\
    s?REFNAME?${refsect}_${dataset}.tif?g" \
    $scriptdir/reg/EM_register.py \
    > $datadir/EM_register.py

qsubfile=$datadir/EM_reg.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo "/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64 --headless $datadir/EM_register.py" >> $qsubfile
sbatch -p compute $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_reg/* /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_reg_arcus/


###====================================###
### convert registered slices to stack ###
###====================================###
#NOTE: using mpi will slow things down terribly here (mem issue)
qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=12" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=s2s" >> $qsubfile
echo "python $scriptdir/convert/EM_series2stack.py \
$datadir/tifs_${dataset}_reg $datadir/${dataset}.h5 \
-f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o" >> $qsubfile
sbatch $qsubfile


###==============================###
### downsample stack for viewing ###
###==============================###
### MEMORY ERROR ON THIS ###
#qsubfile=$datadir/EM_stack2stack_submit.sh
#echo '#!/bin/bash' > $qsubfile
#echo "#SBATCH --nodes=1" >> $qsubfile
#echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
#echo "#SBATCH --time=00:10:00" >> $qsubfile
#echo "#SBATCH --job-name=s2s" >> $qsubfile
#echo "python $scriptdir/convert/EM_stack2stack.py \
#$datadir/${dataset}_reg_tmp.h5 $datadir/${dataset}_reg_tmp_ds.h5 \
#-r 1 10 10 -i zyx -l zyx" >> $qsubfile
#sbatch -p devel $qsubfile

mkdir -p $datadir/${dataset}_reg_ds4

qsubfile=$datadir/EM_downsample_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ds" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_downsample.py \
$datadir/tifs_${dataset}_reg $datadir/tifs_${dataset}_reg_ds4 -d 4 -m" >> $qsubfile
sbatch -p devel $qsubfile

qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=12" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=s2s" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_series2stack.py \
$datadir/tifs_${dataset}_reg_ds4 $datadir/${dataset}_reg_ds4.h5 \
-f 'stack' -z 30 -e 0.0292 0.0292 0.05 -c 20 20 20 -o -m" >> $qsubfile
sbatch -p devel $qsubfile

rm -rf $datadir/tifs_${dataset}_reg_ds4
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_reg_ds4.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/


###===================================###
### apply Ilastik classifier to stack ###
###===================================###

z=0; Z=100;
for x in 1000 2000 3000 4000; do
X=$((x+1000))
qsubfile=$datadir/EM_stack2stack_submit_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}.h5 $datadir/${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}.h5 \
-f 'stack' -g 'stack' -s 20 20 20 -i zyx -l zyx \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z" >> $qsubfile
done
qsub -q develq $qsubfile
done
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_0-*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/

z=0; Z=100;
for x in 0 1000 2000 3000 4000; do
X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
qsubfile=$datadir/EM_ac2s2_${x}-${X}_${y}-${Y}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_lc" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate ilastik-devel" >> $qsubfile
echo "CONDA_ROOT=\`conda info --root\`" >> $qsubfile
echo "\${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/${pixprob_trainingset}_arcus.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs.h5 \
--output_internal_path=/volume/predictions \
$datadir/${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}.h5/stack" >> $qsubfile
sbatch -p compute $qsubfile
# sbatch -p devel $qsubfile
done
done
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_probs.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_probs_arcus.h5
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_0-*_probs.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/

# recombine the _probs.h5 files to m000_probs.h5
qsubfile=$datadir/EM_mb.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $datadir/EM_mergeblocks.py" >> $qsubfile
sbatch -p compute $qsubfile
mkdir -p partials && mv ${dataset}_[0-9]*.h5 ${dataset}_*_probs.h5 partials/


python $scriptdir/convert/EM_mergeblocks.py \
$datadir/${dataset}_probs.h5 '/volume/predictions' 'zyxc' -s 100 4111 4235 6 \
-f $datadir/${dataset}_*-*_*-*_*-*_probs.h5


# create cutouts to evaluate edge effects
z=0; Z=100;
for x in 500 3500; do
X=$((x+1000))
qsubfile=$datadir/EM_stack2stack_submit_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_cu" >> $qsubfile
for y in 500 3500; do
Y=$((y+1000))
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}_probs.h5 $datadir/${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs.h5 \
-f '/volume/predictions' -g '/volume/predictions' -s 20 20 20 6 -i zyxc -l zyxc -e 0.05 0.0073 0.0073 1 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z" >> $qsubfile
done
sbatch -p devel $qsubfile
done
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_500-*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/



# EED on probs  (5GB per 100x1000x1000 block)
#rsync -avz  /Users/michielk/oxscripts/matlab/toolboxes/coherencefilter_version5b/* ndcn0180@arcus.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/oxscripts/matlab/toolboxes/coherencefilter_version5b/
# TODO: write the executable to a different directory (e.g. create bin in $datadir)
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/snippets/eed/EM_eed.m -a /home/ndcn-fmrib-water-brain/ndcn0180/oxscripts/matlab/toolboxes/coherencefilter_version5b
z=0; Z=100;
for layer in 1 2 3; do
for x in 0 1000 2000 3000 4000; do
X=$((x+1000))
qsubfile=$datadir/EM_eed_submit_${x}-${X}_${layer}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
#echo "#SBATCH --ntasks-per-node=3" >> $qsubfile
echo "#SBATCH --time=02:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
echo "$datadir/EM_eed '$datadir' '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs' '/volume/predictions' '/stack' $layer > $datadir/${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs.log &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
done
done
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_4000-5000_4000-5000_0-100_probs?_eed2.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_2000-3000_2000-3000_0-100*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/



# watersheds on EED
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'
python $scriptdir/mesh/prob2labels.py $datadir $dataset -x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100
python $scriptdir/mesh/prob2labels.py $datadir $dataset -x 2000 -X 3000 -y 2000 -Y 3000 -z 0 -Z 100


python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 --MAfile _probs_ws_MA.h5 --MMfile _probs_ws_MM.h5 --UAfile _probs_ws_UA.h5 \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 --MAfile _probs_ws_MA.h5 --MMfile _probs_ws_MM.h5 --UAfile _probs_ws_UA.h5 \
-x 2000 -X 3000 -y 2000 -Y 3000 -z 0 -Z 100


dataset='m000_0-1000_0-1000_0-100'
dataset='m000_2000-3000_2000-3000_0-100'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_seg.h5" \
"${datadir}/${dataset}_seg.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MA.h5" \
"${datadir}/${dataset}_probs_ws_MA.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MM.h5" \
"${datadir}/${dataset}_probs_ws_MM.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_UA.h5" \
"${datadir}/${dataset}_probs_ws_UA.nii.gz" -i 'zyx' -l 'xyz'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_PA.h5" \
"${datadir}/${dataset}_probs_ws_PA.nii.gz" -i 'zyx' -l 'xyz'


python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.tif" -i 'zyx' -l 'xyz'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MAws.h5" \
"${datadir}/${dataset}_probs_ws_MAws.tif" -i 'zyx' -l 'xyz'


# (label2mesh)

# slicvoxels (TODO: adapt commands and prepare for slurm)
qsubfile=$datadir/EM_slic.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_slic" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python $scriptdir/supervoxels/EM_slicvoxels.py \
-i $datadir/$stack.h5 \
-o $datadir/${stack}_slic.h5 \
-f 'stack' -g 'stack' -s 500" >> $qsubfile
echo "python $scriptdir/supervoxels/EM_slicvoxels.py \
-i $datadir/${stack}_probabilities.h5 \
-o $datadir/${stack}_probabilities_slic.h5 \
-f 'probs' -g 'stack' -s 500" >> $qsubfile
qsub -q develq $qsubfile

# agglomeration
# label2mesh


