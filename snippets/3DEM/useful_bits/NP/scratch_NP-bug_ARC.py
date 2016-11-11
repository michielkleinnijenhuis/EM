### EM data MK
# ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
source ~/.bashrc
module load hdf5-parallel/1.8.14_mvapich2_intel
module load mpi4py/1.3.1
module load python/2.7__gcc-4.8
module load matlab/R2015a

# sacct --format=JobID,Timelimit,elapsed,ReqMem,MaxRss,AllocCPUs,AveVMSize,MaxVMSize,CPUTime,ExitCode

# remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP/"
# rsync -avz ~/M3_S1_GNU_NP/train $remdir/
# rsync -avz $remdir/train/orig/*eed2.h5 ~/M3_S1_GNU_NP/train/orig
# rsync -avz $remdir/test/*eed2.h5 ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*_probs.h5 ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/train/*_ws* ~/M3_S1_GNU_NP/train
# rsync -avz $remdir/train/*0300.h5 ~/M3_S1_GNU_NP/train
# rsync -avz $remdir/train/*_prediction_*.h5 ~/M3_S1_GNU_NP/train
# rsync -avz $remdir/test/*0460.h5 ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*_ws* ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*_per* ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*02000*00460.h5 ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*02000*00460_ws*.h5 ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*02000*00460_pred*.h5 ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*_prediction_*.h5 ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*_per* ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/test/*_mask* ~/M3_S1_GNU_NP/test
# rsync -avz $remdir/train/m000_01000-01500_01000-01500_00030-00460_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str1_iter1_parallel_thr0.2_alg1.h5 ~/M3_S1_GNU_NP/train
# rsync -avz $remdir/test/m000_03000-04000_03000-04000_00030-00460_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter2_parallel_thr0.2_alg1.h5 ~/M3_S1_GNU_NP/test

origdir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored
dataset=m000_02000-03000_02000-03000_00030-00460
dataset=m000_03000-04000_03000-04000_00030-00460
cp ${origdir}/${dataset}.h5 $datadir/test
cp ${origdir}/${dataset}_probs.h5 $datadir/test
cp ${origdir}/${dataset}_probs0_eed2.h5 $datadir/test
cp ${origdir}/${dataset}_probs2_eed2.h5 $datadir/test
# cp /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460.h5 /data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP/train/

###=========================================================================###
### Neuroproof ###
###=========================================================================###
CONDA_PATH="$(conda info --root)"
PREFIX="${CONDA_PATH}/envs/neuroproof-test"
NPdir="${HOME}/workspace/Neuroproof_minimal"
datadir="${DATA}/EM/Neuroproof/M3_S1_GNU_NP" && cd $datadir

###========================###
### training agglomeration ###
###========================###
q=''
trainset='train/m000_01000-01500_01000-01500_00030-00460'
# trainset='train/m000_01000-01250_01000-01250_00200-00300'
# trainset='train/m000_01000-01100_01000-01100_00200-00250'
strtype=2
iter=5
svoxpf='_ws_l0.99_u1.00_s005'
# svoxpf='_ws_l0.95_u1.00_s064'
# svoxpf='_slic_s00500_c2.000_o0.050'
gtpf='_PA'
cltype='h5'  # 'h5'  # 'xml' #
classifier="_NPminimal${svoxpf}${gtpf}_str${strtype}_iter${iter}_parallel"
qsubfile=$datadir/NPlearn.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00" || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=NPlearn" >> $qsubfile
echo "export PATH=${CONDA_PATH}/bin:\$PATH" >> $qsubfile
echo "export LD_LIBRARY_PATH=${PREFIX}/lib" >> $qsubfile
echo "$NPdir/NeuroProof_stack_learn \
-watershed $datadir/${trainset}${svoxpf}.h5 stack \
-prediction $datadir/${trainset}_probs.h5 volume/predictions \
-groundtruth $datadir/${trainset}${gtpf}.h5 stack \
-classifier $datadir/${trainset}${classifier}.${cltype} \
-iteration ${iter} -strategy ${strtype} -nomito" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

###========================###
### applying agglomeration ###
###========================###
q=''
# testset=train/m000_01000-01500_01000-01500_00030-00460
# testset=train/m000_01000-01250_01000-01250_00200-00300
# testset=train/m000_01000-01100_01000-01100_00200-00250
testset=test/m000_02000-03000_02000-03000_00030-00460
# testset=test/m000_03000-04000_03000-04000_00030-00460
thr=0.3; alg=1;
qsubfile=$datadir/NPpredict.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00" || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=NPtest" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "export LD_LIBRARY_PATH=${PREFIX}/lib" >> $qsubfile
echo "$NPdir/NeuroProof_stack \
-watershed $datadir/${testset}${svoxpf}.h5 stack \
-prediction $datadir/${testset}_probs.h5 volume/predictions \
-output $datadir/${testset}_prediction${classifier}${cltype}_thr${thr}_alg${alg}M.h5 stack \
-classifier $datadir/${trainset}${classifier}.${cltype} \
-threshold ${thr} -algorithm ${alg}" >> $qsubfile  # -nomito
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile








# export PATH=${CONDA_PATH}/bin:$PATH
# export LD_LIBRARY_PATH=${PREFIX}/lib
# $NPdir/NeuroProof_stack_learn \
# -watershed $datadir/train/${dataset}_slic_s00500_c2.000_o0.050.h5 stack \
# -prediction $datadir/train/${dataset}_probs.h5 volume/predictions \
# -groundtruth $datadir/train/${dataset}_PA.h5 stack \
# -classifier $datadir/train/${dataset}_classifier_str${strtype}_iter${iter}_NPminimal_parallel.h5 \
# -iteration ${iter} -strategy ${strtype} -nomito
