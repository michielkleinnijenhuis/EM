#!/bin/bash

for n in `seq 0 $((njobs-1))`; do

    qsubfile=$datadir/EM_${jobname}_$n.sh

    echo '#!/bin/bash' > $qsubfile
    echo "#PBS -l nodes=$nodes:ppn=$tasks" >> $qsubfile
    echo "#PBS -l mem=$memcpu" >> $qsubfile
    [ "$q" = "d" ] &&
        echo "#PBS -l walltime=00:10:00"  >> $qsubfile ||
            echo "#PBS -l walltime=$wtime" >> $qsubfile
    echo "#PBS -N EM_$jobname" >> $qsubfile
    echo "#PBS -V" >> $qsubfile
    echo "cd \$PBS_O_WORKDIR" >> $qsubfile

    if [[ $additions == *"conda"* ]]
    then
      echo "export PATH=$CONDA_PATH:\$PATH" >> $qsubfile
      echo "source activate $CONDA_ENV" >> $qsubfile
    fi

    if [[ $additions == *"neuroproof"* ]]
    then
        echo "export PATH=$CONDA_PATH:\$PATH" >> $qsubfile
        echo "export LD_LIBRARY_PATH=$CONDA_PATH/envs/$CONDA_ENV/lib" >> $qsubfile
    fi

    datastem=${datastems[n]}
    if [[ $additions == *"mpi"* ]]
    then
        echo ". enable_arcus_mpi.sh" >> $qsubfile
        echo "mpirun \$MPI_HOSTS ${cmd//datastem/$datastem}" >> $qsubfile
    else
        echo "${cmd//datastem/$datastem}" >> $qsubfile
    fi

    [ "$q" = "d" ] &&
        qsub -q develq $qsubfile ||
            qsub $qsubfile

done
