source ~/.bashrc
scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
oddir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/20Mar15/montage/Montage_
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_regpointpairs

mkdir -p $datadir && cd $datadir

module load mpi4py/1.3.1
module load python/2.7__gcc-4.8


###===================================###
### stitch and register slice montage ###
###===================================###

qsubfile=$datadir/EM_reg.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_getpairs.py \
'$datadir/tifs' -o '$datadir/reg_d4_low' \
-t 4 -c 1 -d 4 -k 1000 -f 0.1 0.1 -m -n 50 -r 1" >> $qsubfile
sbatch -p devel $qsubfile

qsubfile=$datadir/EM_reg_optimize_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ro" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_optimize.py \
'$datadir/reg_d4_low' '$datadir/reg_d4_low/betas_c1_d4.npy' \
-r pair_c1_d4_s*.pickle -a 'L-BFGS-B' \
-n 100 -i 100 -m -p 0.628 0.00025 0.00025" >> $qsubfile
sbatch -p devel $qsubfile

qsubfile=$datadir/EM_reg_blend_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=2" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rb" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_blend.py \
'$datadir/tifs' '$datadir/reg_d4_low/betas_c1_d4.npy' '$datadir/reg_d4_low' \
-d 1 -i 1 -m" >> $qsubfile
sbatch -p devel $qsubfile

###===================================###
### stitch and register slice montage ###
###===================================###

qsubfile=$datadir/EM_reg.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=4" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=20:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_getpairs.py \
'$datadir/tifs' -o '$datadir/reg_d4' \
-t 4 -c 1 -d 4 -k 10000 -f 0.1 0.1 -m -n 100 -r 1" >> $qsubfile
sbatch -p compute $qsubfile

qsubfile=$datadir/EM_reg.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_getpairs.py \
'$datadir/tifs' -o '$datadir/reg_d4' \
-t 4 -c 1 -d 4 -k 10000 -f 0.1 0.1 -m -n 100 -r 1 -u $datadir/reg_d4/failed_pairs_c1_d4.pickle" >> $qsubfile
sbatch -p devel $qsubfile

qsubfile=$datadir/EM_reg_optimize_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=4" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ro" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_optimize.py \
'$datadir/reg_d4' '$datadir/reg_d4/betas_c1_d4.npy' \
-r pair_c1_d4_s*.pickle -a 'L-BFGS-B' \
-n 50 -i 400 -m -p 0.628 0.00025 0.00025" >> $qsubfile
sbatch -p compute $qsubfile

qsubfile=$datadir/EM_reg_blend_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=4" >> $qsubfile
echo "#SBATCH --ntasks-per-node=2" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rb" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_blend.py \
'$datadir/tifs' '$datadir/reg_d4/betas_c1_d4.npy' '$datadir/reg_d4' \
-d 1 -i 1 -m" >> $qsubfile
sbatch -p compute $qsubfile


qsubfile=$datadir/EM_reg_blend_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=2" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rb" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/EM_reg_blend.py \
'$datadir/tifs' '$datadir/reg_d4/betas_c1_d4.npy' '$datadir/reg_d4' \
-d 1 -i 1 -m" >> $qsubfile
sbatch -p devel $qsubfile
