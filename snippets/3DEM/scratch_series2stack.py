qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=s2s" >> $qsubfile
echo "python $scriptdir/convert/EM_series2stack.py \
$datadir/${dataset}_reg $datadir/${dataset}_reg_testnonmpi.h5 \
-f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o" >> $qsubfile
sbatch -p devel $qsubfile


mpiexec -n 4 python $scriptdir/convert/EM_series2stack.py $datadir/${dataset}_reg $datadir/${dataset}_reg.h5 -f 'stack' -m -o -e 0.0073 0.0073 0.05 -c 20 20 40


# python $scriptdir/convert/EM_series2stack.py $datadir/${dataset}_reg $datadir/${dataset}_reg.h5 -f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o
# mpiexec -n 4 python $scriptdir/convert/EM_series2stack.py $datadir/${dataset}_reg $datadir/${dataset}_reg.h5 -f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o -m

qsubfile=$datadir/EM_series2stack_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=12" >> $qsubfile
echo "#PBS -l walltime=12:00:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo ". enable_arcus_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/convert/EM_series2stack.py \
$datadir/${dataset}_reg $datadir/${dataset}_reg.h5 \
-f 'stack' -z 30 -e 0.0073 0.0073 0.05 -c 20 20 20 -o -m" >> $qsubfile
qsub $qsubfile
