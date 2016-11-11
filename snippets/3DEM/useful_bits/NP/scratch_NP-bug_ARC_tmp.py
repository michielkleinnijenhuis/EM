datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP'
scriptdir="$HOME/workspace/EM"
dataset=m000
oX=1000;oY=1000;oZ=30;
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
nx=1000; nX=1100; ny=1000; nY=1100; nz=200; nZ=250;
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`
python $scriptdir/convert/EM_stack2stack.py \
$datadir/train/${datastem}_PA.h5 \
$datadir/train/${newstem}_PA.h5 \
-e 0.05 0.0073 0.0073 \
-i 'zyx' -l 'zyx' -d 'int32' \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))
python $scriptdir/convert/EM_stack2stack.py \
$datadir/train/${datastem}_slic_s00500_c2.000_o0.050.h5 \
$datadir/train/${newstem}_slic_s00500_c2.000_o0.050.h5 \
-e 0.05 0.0073 0.0073 \
-i 'zyx' -l 'zyx' -d 'int32' \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))
mv $datadir/train/${newstem}_slic_s00500_c2.000_o0.h5 $datadir/train/${newstem}_slic_s00500_c2.000_o0.050_xyz.h5
python $scriptdir/convert/EM_stack2stack.py \
$datadir/train/${datastem}_probs.h5 \
$datadir/train/${newstem}_probs.h5 \
-f 'volume/predictions' -g 'volume/predictions' \
-e 0.05 0.0073 0.0073 1 \
-i 'zyxc' -l 'zyxc' \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))


NPdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/Neuroproof_minimal
CONDA_PATH=$(conda info --root)
PREFIX=${CONDA_PATH}/envs/neuroproof-test
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP
dataset=m000_01000-01100_01000-01100_00200-00250
strtype=2
iter=5
qsubfile=$datadir/NPlearn.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=NPlearn" >> $qsubfile
echo "export PATH=${CONDA_PATH}/bin:\$PATH" >> $qsubfile
# echo "source activate neuroproof-test" >> $qsubfile
echo "export LD_LIBRARY_PATH=${PREFIX}/lib" >> $qsubfile
echo "$NPdir/NeuroProof_stack_learn \
-watershed $datadir/train/${dataset}_slic_s00500_c2.000_o0.050.h5 stack \
-prediction $datadir/train/${dataset}_probs.h5 volume/predictions \
-groundtruth $datadir/train/${dataset}_PA.h5 stack \
-classifier $datadir/train/${dataset}_classifier_str${strtype}_iter${iter}_NPminimal_parallel.h5 \
-iteration ${iter} -strategy ${strtype} -nomito" >> $qsubfile
sbatch -p devel $qsubfile


export PATH=${CONDA_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${PREFIX}/lib
$NPdir/NeuroProof_stack_learn \
-watershed $datadir/train/${dataset}_slic_s00500_c2.000_o0.050.h5 stack \
-prediction $datadir/train/${dataset}_probs.h5 volume/predictions \
-groundtruth $datadir/train/${dataset}_PA.h5 stack \
-classifier $datadir/train/${dataset}_classifier_str${strtype}_iter${iter}_NPminimal_parallel.h5 \
-iteration ${iter} -strategy ${strtype} -nomito
