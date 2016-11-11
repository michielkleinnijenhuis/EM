# remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/"
# rsync -avz $remdir/restored/m000_*_labelMAmanedit.h5 ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/restored/m000_02000-03000_01000-02000_00030-00460_prediction_*.h5 ~/M3_S1_GNU_NP/test
# remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/"
# rsync -avz $remdir/restored/m000_01000-02000_01000-02000_00030-00460.h5 .
# rsync -avz $remdir/restored/m000_01000-02000_02000-03000_00030-00460.h5 .
# rsync -avz $remdir/restored/m000_02000-03000_01000-02000_00030-00460.h5 .
# rsync -avz $remdir/restored/m000_02000-03000_02000-03000_00030-00460.h5 .

scriptdir="${HOME}/workspace/EM"
datadir="${HOME}/M3_S1_GNU_NP/test/testlab" && cd $datadir
dataset='m000'

pf=''; field='stack'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir/${dataset}${pf}.h5 \
-i `ls $datadir/${dataset}_?????-?????_?????-?????_?????-?????${pf}.h5 | tr "\n" " "` \
-f $field -l 'zyx' -b 1000 1000 30
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${dataset}${pf}.h5 ${datadir}/${dataset}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e 0.0073 0.0073 0.05 -u

pf='_labelMAmanedit'; field='stack'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir/${dataset}${pf}.h5 \
-i `ls $datadir/${dataset}_?????-?????_?????-?????_?????-?????${pf}.h5 | tr "\n" " "` \
-f $field -l 'zyx' -b 1000 1000 30 -r -n
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}${pf}.h5 \
$datadir/${dataset}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'

pf='_ws_l0.95_u1.00_s064_labelMA'; field='stack'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir/${dataset}${pf}.h5 \
-i `ls $datadir/${dataset}_?????-?????_?????-?????_?????-?????${pf}.h5 | tr "\n" " "` \
-f $field -l 'zyx' -b 1000 1000 30 -r -n -m '_maskMA' 'stack'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}${pf}.h5 \
$datadir/${dataset}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'

scriptdir="${HOME}/workspace/EM"
datadir="${HOME}/M3_S1_GNU_NP/test" && cd $datadir
dataset='m000'
pf='_labelMAmanedit'; field='stack'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir/${dataset}${pf}.h5 \
-i `ls $datadir/${dataset}_?????-?????_?????-?????_?????-?????${pf}.h5 | tr "\n" " "` \
-f $field -l 'zyx' -b 0 0 30 -r -n



# convert to nifti
pf="_maskDS"
pf="_labelMA"
scriptdir="${HOME}/workspace/EM"
datadir='/Users/michielk/M3_S1_GNU_NP/test'
dset_name='m000_02000-03000_02000-03000_00030-00460'

pf=_maskDS
pf=_ws_l0.95_u1.00_s064_labelMAold_maskMA
pf=_ws_l0.95_u1.00_s064_labelMAold
pf='_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.1_alg1'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dset_name}${pf}.h5 \
$datadir/${dset_name}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'


###=========================================================================###
### training set
###=========================================================================###
scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/EM/Neuroproof/M3_S1_GNU_NP" && cd $datadir
dataset="m000"
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
### masks for training set
q="d"
qsubfile=$datadir/EM_prob2mask_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=3" >> $qsubfile
echo "#SBATCH --mem-per-cpu=10000" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_p2m" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
${datadir}/train ${datastem} -p \"\" stack -l 0 -u 10000000 -o _maskDS &" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
${datadir}/train ${datastem} -p _probs0_eed2 stack -l 0.2 -o _maskMM &" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
${datadir}/train ${datastem} -p _probs0_eed2 stack -l 0.02 -o _maskMM-0.02 &" >> $qsubfile
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
### watershed on trainingset
q=""
qsubfile=$datadir/EM_supervoxels_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --mem-per-cpu=25000" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=02:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_svox" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "python $scriptdir/supervoxels/EM_watershed.py \
${datadir}/train ${datastem} -c 1 -l 0.99 -u 1 -s 5 &" >> $qsubfile
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
