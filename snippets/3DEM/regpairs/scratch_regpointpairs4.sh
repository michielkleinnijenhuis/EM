scriptdir=~/workspace/EM
datadir=~/oxdata/P01/EM/M3/M3_S1_GNU_stitch/localPP
mkdir -p $datadir && cd $datadir

python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg -t 4 -c 1 -d 4 -k 10000 -f 0.1 0.1 -m -n 100 -r 1
mpiexec -n 8 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg $datadir/reg/betas_o1_d4.npy -r pair*.pickle -a 'L-BFGS-B' -m
python $scriptdir/reg/EM_reg_blend.py $datadir/tifs $datadir/reg/betas_o1_d4.npy $datadir/reg -d 4 -i 1 -m

mpiexec -n 4 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg_mpi -t 4 -c 1 -d 4 -k 10000 -f 0.1 0.1 -m -n 100 -r 1
mpiexec -n 8 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_mpi $datadir/reg_mpi/betas_o1_d4.npy -r pair*.pickle -a 'L-BFGS-B' -m
python $scriptdir/reg/EM_reg_blend.py $datadir/tifs $datadir/reg_mpi/betas_o1_d4.npy $datadir/reg_mpi -d 4 -i 1 -m

mpiexec -n 4 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg_mpi_p -t 4 -c 1 -d 4 -k 10000 -f 0.1 0.1 -m -n 100 -r 1 -p
mpiexec -n 8 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_mpi_p $datadir/reg_mpi_p/betas_o1_d4.npy -r pair*.pickle -a 'L-BFGS-B' -m
python $scriptdir/reg/EM_reg_blend.py $datadir/tifs $datadir/reg_mpi_p/betas_o1_d4.npy $datadir/reg_mpi_p -d 4 -i 1 -m

mpiexec -n 8 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg_d1 -t 4 -c 1 -d 1 -k 10000 -f 0.1 0.1 -m -n 100 -r 1 -p
mpiexec -n 8 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_d1 $datadir/reg_d1/betas_o1_d1.npy -r pair*.pickle -a 'L-BFGS-B' -m
python $scriptdir/reg/EM_reg_blend.py $datadir/tifs $datadir/reg_d1/betas_o1_d1.npy $datadir/reg_d1 -d 1 -i 1 -m







scriptdir=~/workspace/EM
datadir=~/oxdata/P01/EM/M3/M3_S1_GNU_regpointpairs
mkdir -p $datadir && cd $datadir

mpiexec -n 8 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg_d4 -t 4 -c 1 -d 4 -k 10000 -f 0.1 0.1 -m -n 100 -r 1 -u $datadir/reg_d4/failed_pairs_c1_d4.pickle
mpiexec -n 8 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_d4 $datadir/reg_d4/betas_o1_d4.npy -r pair_c1_d4_s*.pickle -a 'L-BFGS-B' -m
mpiexec -n 8 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_d4 $datadir/reg_d4/betas_o1_d4.npy -r pair_c1_d4_s*.pickle -a 'L-BFGS-B' -n 20 -i 20 -m

python $scriptdir/reg/EM_reg_blend.py $datadir/tifs $datadir/reg_d4/betas_o1_d4.npy $datadir/reg_d4 -d 4 -i 1 -m


mpiexec -n 6 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg_d1 -t 4 -c 1 -d 1 -k 10000 -f 0.1 0.1 -m -n 100 -r 1 -p

python $scriptdir/reg/EM_reg_blend.py $datadir/tifs $datadir/reg_d4/betas_o1_d4.npy $datadir/reg_d1 -d 4 -o 1 -i 1 -m

(n_slcs - 1) * 14 + 5





mpiexec -n 6 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs_10slcs -o $datadir/reg_10slcs_d4 -t 4 -c 1 -d 4 -k 10000 -f 0.1 0.1 -m -n 100 -r 1 -p
mpiexec -n 6 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_10slcs_d4 $datadir/reg_10slcs_d4/betas_o1_d4.npy -r pair_c1_d4_s*.pickle -a 'L-BFGS-B' -n 1000 -i 1000 -m
python $scriptdir/reg/EM_reg_blend.py $datadir/tifs_10slcs $datadir/reg_10slcs_d4/betas_o1_d4.npy $datadir/reg_10slcs_d4 -d 1 -i 1 -m


mpiexec -n 6 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_10slcs_d4 $datadir/reg_10slcs_d4/betas_o1_d4_pc.npy -r pair_c1_d4_s*.pickle -a 'L-BFGS-B' -n 20 -i 100 -m -p 0.628 0.00025 0.00025
python $scriptdir/reg/EM_reg_blend.py $datadir/tifs_10slcs $datadir/reg_10slcs_d4/betas_o1_d4_pc.npy $datadir/reg_10slcs_d4 -d 1 -i 1 -m




# with EuclidianTransform
conda create --name scikit-image-devel_0.13 python=3.5
source activate scikit-image-devel_0.13
conda install numpy
pip install git+git://github.com/scikit-image/scikit-image@master
python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg -t 4 -c 1 -d 4 -k 10000 -f 0.1 0.1 -n 100 -r 1 -p
python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg_1000 -t 4 -c 1 -d 4 -k 1000 -f 0.1 0.1 -n 50 -r 1 -p
# FAIL: mpiexec -n 4 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg_mpi -t 4 -c 1 -d 4 -k 1000 -f 0.1 0.1 -m -n 50 -r 1 -p

# with other mpi4py binaries (and python2.7 [not required?]
conda create --name scikit-image-devel_0.13_p2 python=2.7
source activate scikit-image-devel_0.13_p2
conda install numpy
conda install --channel mpi4py mpich mpi4py  #https://groups.google.com/forum/#!topic/mpi4py/ULMq-bC1oQA
pip install git+git://github.com/scikit-image/scikit-image@master
mpiexec -n 6 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs -o $datadir/reg_mpi -t 4 -c 1 -d 4 -k 1000 -f 0.1 0.1 -n 50 -r 1 -p
mpiexec -n 6 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_mpi $datadir/reg_mpi/betas_o1_d4.npy -r pair_c1_d4_s*.pickle -a 'L-BFGS-B' -n 50 -i 200 -m
python $scriptdir/reg/EM_reg_blend.py $datadir/tifs $datadir/reg_mpi/betas_o1_d4.npy $datadir/reg_mpi -d 1 -i 1


mpiexec -n 6 python $scriptdir/reg/EM_reg_getpairs.py $datadir/tifs $datadir/reg_mpi_test \
-t 4 -o 1 -d 4 -k 1000 -f 0.1 0.1 -n 50 -r 1 -w SimilarityTransform -c $datadir/connectivities.txt



# ref to arbitrary slice test
source activate scikit-image-devel_0.13_p2
scriptdir=~/workspace/EM
datadir=~/oxdata/P01/EM/M3/M3_S1_GNU_stitch/localPP
mkdir -p $datadir && cd $datadir
mpiexec -n 6 python $scriptdir/reg/EM_reg_optimize.py $datadir/reg_mpi $datadir/reg_mpi/betas_o1_d4_as.npy -r pair_c1_d4_s*.pickle -a 'L-BFGS-B' -n 50 -i 200 -m -f 4 3
