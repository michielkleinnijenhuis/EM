source ~/.bashrc
scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_stitch
x=0
X=4000
y=0
Y=4000
z=0
Z=460

mkdir -p $datadir && cd $datadir

module load mpi4py/1.3.1
module load python/2.7__gcc-4.8

mkdir -p $datadir/tifs
for no in `seq -f '%04g' 0 6`; do
cp $datadir/../M3_S1_GNU/tifs/$no_*.tif $datadir/tifs
done

qsubfile=$datadir/EM_reg.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_getpairs.py \
'$datadir/tifs' -o '$datadir/reg_d4' \
-t 4 -c 1 -d 4 -k 50000 -f 0.1 0.1 -m -n 300 -r 1" >> $qsubfile
sbatch -p devel $qsubfile

python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg -t 4 -c 1 -d 1 -k 10000 -f 0.1 0.1 -m -n 100 -r 1



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
