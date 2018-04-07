

function template_job_array {
    # Generate ARC submission files and submit jobs to Arcus-B.
    # Series of commands per submission file for array processing.

    serial=$1

    JOBS=()

    for n in `seq 0 $((njobs-1))`; do

        subfile=$datadir/EM_${jobname}_`printf %03d $n`.sh
        echo '#!/bin/bash' > $subfile

        sbatch_directives
        conda_directives
        neuroproof_directives

        for t in `seq 0 $((tasks-1))`; do

            datastem=${datastems[n*tasks+t]}
            [ -z "$datastem" ] && continue

            cmdt=$cmd
            cmd_deletelabels
            cmd_mergelabels
            cmd_replace_datastem $datastem

            [ "$serial" != 'serial' ] &&
                { echo "$cmdt &" >> $subfile ; } ||
                    { echo "$cmdt" >> $subfile ; }

        done

        [ "$serial" != 'serial' ] && echo "wait" >> $subfile

        submit_job

        JOBS+=${JOB##* }

    done

    export JOBS
}
