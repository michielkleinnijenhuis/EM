###=======================###
### watershed supervoxels ###
###=======================###
### local
scriptdir="$HOME/workspace/EM"
datadir='/Users/michielk/M3_S1_GNU_NP/train/orig'
datastem='m000_01000-01500_01000-01500_00030-00460'
datastem='m000_01000-01250_01000-01250_00200-00300'
# on data
python $scriptdir/supervoxels/EM_watershed.py \
${datadir} ${datastem} -l 30000 -u 100000 -s 64
# on prob_ics
python $scriptdir/supervoxels/EM_watershed.py \
${datadir} ${datastem} -l 0.99 -u 1 -s 5
python $scriptdir/supervoxels/EM_watershed.py \
${datadir} ${datastem} -l 0.95 -u 1 -s 64


### ARC
origdir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored
dataset=m000_03000-04000_03000-04000_00030-00460
cp ${origdir}/${dataset}.h5 $datadir/test
cp ${origdir}/${dataset}_probs.h5 $datadir/test
cp ${origdir}/${dataset}_probs0_eed2.h5 $datadir/test
cp ${origdir}/${dataset}_probs2_eed2.h5 $datadir/test

scriptdir="$HOME/workspace/EM"
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP
cd $datadir

ddir=$datadir/train; datastem=m000_01000-01500_01000-01500_00030-00460
ddir=$datadir/train; datastem=m000_01000-01250_01000-01250_00200-00300
ddir=$datadir/train; datastem=m000_01000-01100_01000-01100_00200-00250
ddir=$datadir/test; datastem=m000_02000-03000_02000-03000_00030-00460
ddir=$datadir/test; datastem=m000_03000-04000_03000-04000_00030-00460
q='d'
qsubfile=$datadir/svox.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00" || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=svox" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
# echo "source activate scikit-image-devel_p34" >> $qsubfile
echo "python $scriptdir/supervoxels/EM_watershed.py \
${ddir} ${datastem} -l 0.95 -u 1 -s 64" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile



###==============================###
### watershed supervoxels (gala) ###
###==============================###

# source activate gala_20160715
# for dataset in 'training_sample1' 'training_sample2' 'validation_sample'; do
# gala-segmentation-pipeline \
# --image-stack $datadir/$dataset/grayscale_maps.h5 \
# --ilp-file $datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
# --disable-gen-pixel \
# --pixelprob-file $datadir/$dataset/boundary_prediction.h5 \
# --enable-gen-supervoxels \
# --disable-gen-agglomeration \
# --seed-size 5 \
# --enable-raveler-output \
# --enable-h5-output $datadir/$dataset \
# --segmentation-thresholds 0.0
# done



###==================###
### slic supervoxels ###
###==================###
### slic on test data (int32 vs int64?)  # 45 min; 25g
# ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
source ~/.bashrc
module load hdf5-parallel/1.8.14_mvapich2_intel
scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/Neuroproof/M3_S1_GNU_NP" && cd $datadir
dataset='m000'
nvox=500; comp=2; smooth=0.05;  #nvox=500; comp=0.02; smooth=0.01;
qsubfile=$datadir/EM_slic_s${nvox}_c${comp}_o${smooth}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_slic" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_p34" >> $qsubfile
xs=1000; ys=1000;
z=30; Z=460;
for x in 3000; do
for y in 3000; do
[ $x == 5000 ] && X=5217 || X=$((x+xs))
[ $y == 4000 ] && Y=4460 || Y=$((y+ys))
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
echo "python $scriptdir/supervoxels/EM_slicvoxels.py \
${datadir}/test/${datastem}.h5 \
${datadir}/test/${datastem}_slic_s`printf %05d ${nvox}`_c`printf %.3f ${comp}`_o`printf %.3f ${smooth}`.h5 \
-f 'stack' -g 'stack' -s ${nvox} -c ${comp} -o ${smooth} -u \
> EM_slic_${datastem} &" >> $qsubfile
done
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile

datastem=test/m000_03000-04000_03000-04000_00030-00460
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}_slic_s00500_c2.000_o0.050.h5 \
$datadir/${datastem}_slic_s00500_c2.000_o0.050.h5 \
-e 0.05 0.0073 0.0073 \
-i 'zyx' -l 'zyx' -d 'int32'
mv $datadir/${datastem}_slic_s00500_c2.000_o0.h5 $datadir/${datastem}_slic_s00500_c2.000_o0.050.h5
