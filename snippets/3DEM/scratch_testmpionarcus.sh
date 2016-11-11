module load python/2.7
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2

export template='single-pbs' additions='mpi'
export njobs=1 nodes=2 tasks=16 memcpu=60GB wtime="00:10:00" q="d"
export jobname="cc2Dfilter"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dfilter' -m \
-d 0 --maskMB '_maskMB' 'stack' \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw_test" 'stack' \
-a 10 -A 1500 -e 0 -x 0.30 -s 0.50 \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh
# 410417

module load python/2.7
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2
export template='single-pbs' additions='mpi'
export njobs=1 nodes=4 tasks=16 memcpu=60GB wtime="01:00:00" q="d"
export jobname="ccmerge1"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem -M 'MAstitch' -m \
-l "${CC2Dstem}_fw_3Dlabeled" 'stack' -o "${CC2Dstem}_fw_3Diter1" 'stack' \
-d 0 -r 4 -t 0.50 > $datadir/test.log"
source $scriptdir/pipelines/template_job_$template.sh



# arcus-b fails with parallel h5py
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2_gcc
module load python/2.7__gcc-4.8

export template='single' additions='mpi'
export njobs=1 nodes=1 tasks=2 memcpu=60000 wtime="01:00:00" q=""
export jobname="cc2DfilterPBS"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dfilter' \
-d 0 --maskMB '_maskMB' 'stack' \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw" 'stack' \
-a 10 -A 1500 -e 0 -x 0.30 -s 0.50 \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh
