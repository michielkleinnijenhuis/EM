source ~/.bashrc
scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/archive/reg_pointpairs
x=0
X=4000
y=0
Y=4000
z=0
Z=460

mkdir -p $datadir && cd $datadir

module load mpi4py/1.3.1
module load python/2.7__gcc-4.8



mkdir -p $datadir/reg_d4_tmp
for no in `seq -f '%04g' 0 10`; do
cp $datadir/reg_d4/pair_c1_d4_s${no}-t?_s${no}-t?.pickle $datadir/reg_d4_tmp
done


qsubfile=$datadir/EM_reg_optimize_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ro" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_optimize.py \
'$datadir/reg_d4_tmp' '$datadir/reg_d4_tmp/betas_o1_d4.npy' -m" >> $qsubfile
sbatch -p devel $qsubfile

qsubfile=$datadir/EM_reg_blend_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=8" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rb" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_blend.py \
'$datadir/tifs' '$datadir/reg_d4_tmp/betas_o1_d4.npy' '$datadir/reg_d4_tmp' \
-d 4 -i 1 -m" >> $qsubfile
sbatch -p compute $qsubfile
