#!/bin/bash

for n in `seq 0 $((nodes-1))`; do

    qsubfile=$datadir/EM_${jobname}_$n.sh

    echo '#!/bin/bash' > $qsubfile
    echo "#SBATCH --nodes=1" >> $qsubfile
    echo "#SBATCH --ntasks-per-node=$tasks" >> $qsubfile
    echo "#SBATCH --mem-per-cpu=$memcpu" >> $qsubfile
    [ "$q" = "d" ] &&
        echo "#SBATCH --time=00:10:00"  >> $qsubfile ||
            echo "#SBATCH --time=$wtime" >> $qsubfile
    echo "#SBATCH --job-name=EM_$jobname" >> $qsubfile

    for t in `seq 0 $((tasks-1))`; do
        datastem=${datastems[n*tasks+t]}
        dellabels=`grep $datastem $editsfile | awk '{$1 = ""; print $0;}'`
        tmpcmd=${cmd//datastem/$datastem}
        echo "${tmpcmd//dellabels/$dellabels} &" >> $qsubfile
    done

    echo "wait" >> $qsubfile

    [ "$q" = "d" ] &&
        sbatch -p devel $qsubfile ||
            sbatch $qsubfile

done
