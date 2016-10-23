#!/bin/bash

for n in `seq 0 $((njobs-1))`; do

    qsubfile=$datadir/EM_${jobname}_$n.sh

    echo '#!/bin/bash' > $qsubfile
    echo "#SBATCH --nodes=$nodes" >> $qsubfile
    echo "#SBATCH --ntasks-per-node=$tasks" >> $qsubfile
    echo "#SBATCH --mem-per-cpu=$memcpu" >> $qsubfile
    [ "$q" = "d" ] &&
        echo "#SBATCH --time=00:10:00"  >> $qsubfile ||
            echo "#SBATCH --time=$wtime" >> $qsubfile
    echo "#SBATCH --job-name=EM_$jobname" >> $qsubfile

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
        echo ". enable_arcus-b_mpi.sh" >> $qsubfile
        echo "mpirun \$MPI_HOSTS ${cmd//datastem/$datastem}" >> $qsubfile
    else
        echo "${cmd//datastem/$datastem}" >> $qsubfile
    fi

    [ "$q" = "d" ] &&
        sbatch -p devel $qsubfile ||
            sbatch $qsubfile

done

