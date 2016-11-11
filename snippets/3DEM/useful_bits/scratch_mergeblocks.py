scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
dataset='m000'

pf=
pf='_probs_ws_MA'
python $scriptdir/convert/EM_mergeblocks.py \
-i "$datadir/${dataset}_01000-02000_01000-02000_00030-00460${pf}.h5" \
"$datadir/${dataset}_01000-02000_02000-03000_00030-00460${pf}.h5" \
-o "$datadir/${dataset}_01000-02000_01000-03000_00030-00460${pf}.h5" \
-f 'stack' -l 'zyx' -b 1000 1000 30
python $scriptdir/convert/EM_mergeblocks.py \
-i "$datadir/${dataset}_01000-02000_01000-02000_00030-00460_probs_ws_MAfilled.h5" \
"$datadir/${dataset}_01000-02000_02000-03000_00030-00460_probs_ws_MA_probs_ws_MAfilled.h5" \
"$datadir/${dataset}_02000-03000_01000-02000_00030-00460_probs_ws_MA_probs_ws_MAfilled.h5" \
"$datadir/${dataset}_02000-03000_02000-03000_00030-00460_probs_ws_MA_probs_ws_MAfilled.h5" \
-o "$datadir/${dataset}_01000-02000_01000-03000_00030-00460_probs_ws_MA.h5" \
-f 'stack' -l 'zyx' -b 1000 1000 30
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_01000-02000_01000-03000_00030-00460${pf}.h5" \
"${datadir}/${dataset}_01000-02000_01000-03000_00030-00460${pf}.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

python $scriptdir/convert/EM_mergeblocks.py \
-i "$datadir/${dataset}_01000-02000_01000-02000_00030-00460_probs_ws_MAfilled.h5" \
"$datadir/${dataset}_01000-02000_02000-03000_00030-00460_probs_ws_MA_probs_ws_MAfilled.h5" \
-o "$datadir/${dataset}_01000-03000_01000-03000_00030-00460_probs_ws_MA.h5" \
-f 'stack' -l 'zyx' -b 1000 1000 30
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_01000-03000_01000-03000_00030-00460${pf}.h5" \
"${datadir}/${dataset}_01000-03000_01000-03000_00030-00460${pf}.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

python $scriptdir/convert/EM_mergeblocks.py \
-i "$datadir/${dataset}_01000-02000_01000-02000_00030-00460.h5" \
"$datadir/${dataset}_01000-02000_02000-03000_00030-00460.h5" \
"$datadir/${dataset}_02000-03000_01000-02000_00030-00460.h5" \
"$datadir/${dataset}_02000-03000_02000-03000_00030-00460.h5" \
-o "$datadir/${dataset}_01000-03000_01000-03000_00030-00460.h5" \
-f 'stack' -l 'zyx' -b 1000 1000 30
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_01000-03000_01000-03000_00030-00460.h5" \
"${datadir}/${dataset}_01000-03000_01000-03000_00030-00460.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u




### 2x2 sections processing
# for the raw datafiles, the _ is necessary due to bug in EM_mergeblocks.py
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


### merge a 2x2 subset of '_probs' blocks and extract prob0
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




### merge a 9x9 block

scriptdir="$HOME/workspace/EM"
# DATA="$HOME/oxdata/P01"
datadir="$DATA/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'

# pf=; field='stack'
# pf=_probs; field='volume/predictions'
pf=_probs0_eed2; field='stack'
qsubfile=$datadir/EM_mb${pf}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=02:00:00" >> $qsubfile
echo "#SBATCH --mem=250000" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py -i \
$datadir/${dataset}_00000-01000_00000-01000_00000-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_00000-01000_00000-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_00000-01000_00000-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_01000-02000_00000-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_01000-02000_00000-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_01000-02000_00000-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_02000-03000_00000-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_02000-03000_00000-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_02000-03000_00000-00460${pf}.h5 \
-o $datadir/${dataset}_00000-03000_00000-03000_00000-00430${pf}.h5 \
-f $field -l 'zyx'" >> $qsubfile
sbatch -p compute $qsubfile
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_00000-03000_?????-?????_?????-?????.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
pf=;
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_00000-03000_00000-03000_00000-00430.h5" \
"${datadir}/${dataset}_00000-03000_00000-03000_00000-00430.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u




### merge everything
python $scriptdir/convert/EM_mergeblocks.py \
-i $datadir/${dataset}_*-*_*-*_*-*_probs.h5 \
-f 'volume/predictions' -e 0.05 0.0073 0.0073 1 -l 'zyxc'

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


### merge everything for the _eed outputs
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


### merge all blocks (explicit)
pf=; field='stack'
# pf=_probs; field='volume/predictions'
pf=_probs0_eed2; field='stack'
qsubfile=$datadir/EM_mb${pf}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=02:00:00" >> $qsubfile
#echo "#SBATCH --mem=50000" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py -i \
$datadir/${dataset}_00000-01000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_04000-04460_00030-00460${pf}.h5 \
-o $datadir/${dataset}_00000-05217_00000-04460_00030-00460${pf}.h5 \
-f $field -l 'zyx'" >> $qsubfile
sbatch -p compute $qsubfile
