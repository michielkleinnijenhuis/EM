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

    for t in `seq 0 $((tasks-1))`; do
        datastem=${datastems[n*tasks+t]}

        if [[ $additions == *"dellabels"* ]]
        then
            dellabels=`grep $datastem $editsfile | awk '{$1 = ""; print $0;}'`
            echo $dellabels
            tmpcmd=${cmd//datastem/$datastem}
            echo "${tmpcmd//dellabels/$dellabels} &" >> $qsubfile
        else
            echo "${cmd//datastem/$datastem} &" >> $qsubfile
        fi

    done

    echo "wait" >> $qsubfile

    [ "$q" = "d" ] &&
        sbatch -p devel $qsubfile ||
            sbatch $qsubfile

done

