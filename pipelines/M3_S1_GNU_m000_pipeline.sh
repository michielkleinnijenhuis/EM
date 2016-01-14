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
datadir="$DATA/EM/M3/M3_S1_GNU" && cd $datadir
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
z=0; Z=430;
for x in 0 1000 2000 3000 4000; do
X=$((x+1000))
qsubfile=$datadir/EM_stack2stack_submit_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}.h5 \
$datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`.h5 \
-f 'stack' -g 'stack' -s 20 20 20 -i zyx -l zyx \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z" >> $qsubfile
done
sbatch -p devel $qsubfile
done
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_0-*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
# FIXME: naming goes wrong for $Z

z=0; Z=430;
for x in 0 1000 2000 3000 4000 5000; do
X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
qsubfile=$datadir/EM_ac2s2_${x}-${X}_${y}-${Y}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_il_${x}-${X}_${y}-${Y}" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate ilastik-devel" >> $qsubfile
echo "CONDA_ROOT=\`conda info --root\`" >> $qsubfile
echo "\${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/${pixprob_trainingset}_arcus.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs.h5 \
--output_internal_path=/volume/predictions \
$datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`.h5/stack" >> $qsubfile
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
echo "python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_*-*_*-*_*-*_probs.h5 \
-f 'volume/predictions' -e 0.05 0.0073 0.0073 1 -l 'zyxc'" >> $qsubfile
sbatch -p compute $qsubfile
mkdir -p partials && mv $datadir/${dataset}_*-*_*-*_*-*.h5 partials/


# create cutouts to evaluate edge effects
### edge effect occur from applying Ilastik classifier
### they also appear to be more prominent in eed2

###=======================###
### EED on probabilities  ###  (5GB per 100x1000x1000 block) (20GB per 430x1000x1000 block: +2min per iteration)
###=======================###
#rsync -avz  /Users/michielk/oxscripts/matlab/toolboxes/coherencefilter_version5b/* ndcn0180@arcus.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/oxscripts/matlab/toolboxes/coherencefilter_version5b/
mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/snippets/eed/EM_eed.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b
z=0; Z=460;
for x in 0 1000 2000 3000 4000; do
X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
qsubfile=$datadir/EM_eed_submit_${x}-${X}_${y}-${Y}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=3" >> $qsubfile
echo "#SBATCH --time=05:00:00" >> $qsubfile
echo "#SBATCH --mem=100000" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
for layer in 1 2 3; do
[ -f $datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs$((layer-1))_eed2.h5 ] || {
echo "$datadir/bin/EM_eed '$datadir' \
'${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs' \
'/volume/predictions' '/stack' $layer \
> $datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs.log &" >> $qsubfile ; }
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
done
done
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_4000-5000_4000-5000_0-100_probs?_eed2.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_2000-3000_2000-3000_0-100*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/


###===================###
### watersheds on EED ###
###===================###
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/0250_m000_seg.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/archive/m000_z0000-z0100/
z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
#[ $x == 4000 ] && X=4235 || X=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --time=100:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
#[ $y == 4000 ] && Y=4111 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
done

z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
#[ $x == 4000 ] && X=4235 || X=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --time=100:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
#[ $y == 4000 ] && Y=4111 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' --MAsegfile '_probs_ws_MA' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
done

z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
#[ $x == 4000 ] && X=4235 || X=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --time=100:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
#[ $y == 4000 ] && Y=4111 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' --MAfile '_probs_ws_MA_probs_ws_MAfilled' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
done

###=======================###
### merge the labelimages ###(_probs_ws_PA; _probs_ws_MAfilled)
###=======================###
z=30; Z=460;
qsubfile=$datadir/EM_mb.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=2" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for pf in _probs_ws_MAfilled _probs_ws_PA; do
echo "python $scriptdir/convert/EM_mergeblocks.py -i \\" >> $qsubfile
for x in 1000 2000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 1000 2000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "$datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`${pf}.h5 \\" >> $qsubfile
done
done
echo "-o $datadir/${dataset}_01000-03000_01000-03000_00030-00460${pf}.h5 \\" >> $qsubfile
echo "-f 'stack' -l 'zyx' -b 1000 1000 30 &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile


###============================###
### convert labelimages to stl ###
###============================###
x=1000; X=3000; y=1000; Y=3000; z=30; Z=460;
qsubfile=$datadir/EM_l2s.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_l2s" >> $qsubfile
echo "python $scriptdir/mesh/label2stl.py $datadir $dataset \
-L '_probs_ws_MAfilled' '_probs_ws_PA' -c 'MA' 'PA' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z" >> $qsubfile
sbatch -p devel $qsubfile

# TODO: labelclass names, memoryfootprint (about 15GB per 460x1000x1000 block), parallelize?, mirror labelimages in z?

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
dataset='m000'
x=1000; X=2000; y=1000; Y=2000; z=30; Z=460;
python $scriptdir/mesh/label2stl.py $datadir $dataset \
-L '_probs_ws_MAfilled' '_probs_ws_PA' -c 'MA' 'PA' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z
python $scriptdir/mesh/label2stl.py $datadir $dataset \
-L '_probs_ws_MAfilled' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z
python $scriptdir/mesh/label2stl.py $datadir $dataset \
-L '_probs_ws_PA' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

###========================###
### convert stl to blender ###
###========================###
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf UA -L 'UA' -e 0.1 -s 0.5 10 -d 0.2
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf ECS -L 'ECS' -e -0.2 -s 0.5 10 -d 0.2

###============================###
### convert stl to MCell model ###
###============================###





 # --MMfile '_probs_ws_MM.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5'


scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
-x 2000 -X 3000 -y 2000 -Y 3000 -z 0 -Z 100



python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 --MAfile _probs_ws_MA.h5 --MMfile _probs_ws_MM.h5 --UAfile _probs_ws_UA.h5 \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 --MAfile _probs_ws_MA.h5 --MMfile _probs_ws_MM.h5 --UAfile _probs_ws_UA.h5 \
-x 2000 -X 3000 -y 2000 -Y 3000 -z 0 -Z 100


dataset='m000_0-1000_0-1000_0-100'
#dataset='m000_2000-3000_2000-3000_0-100'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_seg.h5" \
"${datadir}/${dataset}_seg.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MA.h5" \
"${datadir}/${dataset}_probs_ws_MA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MM.h5" \
"${datadir}/${dataset}_probs_ws_MM.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_UA.h5" \
"${datadir}/${dataset}_probs_ws_UA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_PA.h5" \
"${datadir}/${dataset}_probs_ws_PA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05


python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.tif" -i 'zyx' -l 'xyz'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MAws.h5" \
"${datadir}/${dataset}_probs_ws_MAws.tif" -i 'zyx' -l 'xyz'


# (label2mesh)

# slicvoxels (TODO: adapt commands and prepare for slurm)

nvox=500
for comp in 0.01 0.1 1 10 100; do
smooth=0
qsubfile=$datadir/EM_slic_s${nvox}_c${comp}_o${smooth}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=10:00:00" >> $qsubfile
#echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_slic" >> $qsubfile
x=1000; X=2000; y=1000; Y=2000; z=30; Z=460;
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
echo "python $scriptdir/supervoxels/EM_slicvoxels.py \
${datadir}/${datastem}.h5 ${datadir}/${datastem}_slic_s${nvox}_c${comp}_o${smooth}.h5 \
-f 'stack' -g 'stack' -s ${nvox} -c ${comp} -o ${smooth} -u" >> $qsubfile
sbatch -p compute $qsubfile
done

x=1000; X=2000; y=1000; Y=2000; z=30; Z=460;
qsubfile=$datadir/EM_slic.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_slic" >> $qsubfile
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
echo "python $scriptdir/supervoxels/EM_slicvoxels.py \
${datadir}/${datastem}_probs.h5 ${datadir}/${datastem}_probs_slic.h5 \
-f 'volume/predictions' -g 'stack' -s 500 -c 0.2 -o 1 -e 0.05 0.0073 0.0073" >> $qsubfile
sbatch -p compute $qsubfile


# agglomeration
# label2mesh


