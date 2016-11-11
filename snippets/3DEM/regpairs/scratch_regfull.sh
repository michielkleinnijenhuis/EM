source ~/.bashrc
scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
oddir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/20Mar15/montage/Montage_
datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU

mkdir -p $datadir && cd $datadir

# module load mpi4py/1.3.1
module load mvapich2/2.0.1__intel-2013
module load python/3.4__gcc-4.8

###===================================###
### stitch and register slice montage ###
###===================================###

q=
qsubfile=$datadir/EM_reg.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=4" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/get_pairs.py \
$datadir/tifs $datadir/reg_d4 -c $datadir/connectivities.txt \
-d 4 -m -n 100 -r 1" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

# missing pairs
q="d"
qsubfile=$datadir/EM_reg.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/get_pairs.py \
$datadir/tifs $datadir/reg_d4 -c $datadir/connectivities.txt \
-d 4 -m -n 100 -r 1 -u $datadir/reg_d4/failed_pairs_c1_d4.pickle" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

# failed pairs
q="d"
qsubfile=$datadir/EM_reg.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rs" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/get_pairs.py \
$datadir/tifs $datadir/reg_d4 -c $datadir/connectivities.txt \
-d 4 -m -k 100000 -n 100 -r 3 -u $datadir/reg_d4/failed_pairs_c1_d4.pickle" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

q=""
qsubfile=$datadir/EM_reg_optimize_submit.sh
echo '#!/bin/bash' > $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --nodes=2"  >> $qsubfile || echo "#SBATCH --nodes=32" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=40:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ro" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/optimize_transforms.py \
$datadir/reg_d4 $datadir/reg_d4/betas_c1_d4.npy \
-r pair_c1_d4_s*.pickle -a 'L-BFGS-B' \
-n 50 -i 500 -m -p 0.628 0.00025 0.00025" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

q="d"
qsubfile=$datadir/EM_reg_blend_submit.sh
echo '#!/bin/bash' > $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --nodes=2"  >> $qsubfile || echo "#SBATCH --nodes=16" >> $qsubfile
echo "#SBATCH --ntasks-per-node=2" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rb" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/blend_tiles.py \
$datadir/tifs $datadir/reg_d4/betas_c1_d4.npy $datadir/reg_d4 \
-d 1 -i 1 -f 250 0 -m" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

# source activate scikit-image-devel_0.13
# pairfile = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4_tmp/pair_c1_d4_s0437-t0_s0438-t3.pickle'
# p, src, dst, model, w = pickle.load(open(pairfile, 'rb'), encoding='latin1')



### test subset of pairs
q="d"
qsubfile=$datadir/EM_reg_optimize_submit.sh
echo '#!/bin/bash' > $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --nodes=1"  >> $qsubfile || echo "#SBATCH --nodes=16" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ro" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/optimize_transforms.py \
$datadir/reg_d4_test $datadir/reg_d4_test/betas_c1_d4.npy \
-r pair_c1_d4_s*.pickle -a 'L-BFGS-B' \
-n 50 -i 400 -m -p 0.628 0.00025 0.00025" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile

qsubfile=$datadir/EM_reg_blend_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=2" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_rb" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/reg/blend_tiles.py \
$datadir/tifs_test $datadir/reg_d4_test/betas_c1_d4.npy $datadir/reg_d4_test \
-d 1 -i 1 -f 0 0 -m" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
