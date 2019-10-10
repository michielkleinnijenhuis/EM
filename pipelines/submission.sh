#!/bin/bash


function single_job {
    # Generate scripts and submit/execute jobs.

    local cmd="$*"
    local subfile

    cmd="$( cmd_prefix_mpi "$cmd" )"

    subfile=$( generate_job_script "$cmd" )
    chmod u+x $subfile

    if [ "$compute_env" == "LOCAL" ]  # Process locally.
    then

        [ "$q" = "h" ] &&
            echo "$subfile" ||
                $subfile

    else  # Submit to cluster.

        jid=$( submit_job "$subfile" )  # slurm/qsub answer or scriptpath
        jid=${jid##* }  # retain only first word (jobID or scriptpath)
        echo "$jid"

    fi

}


function array_job {
    # Submit a series of jobs.
        # THe number of jobs is specified through arg1.
        # The number of commands per job is specified through arg2.
        # Job commands are selected from the strings in $cmdarray.

    local njobs="$1"
    local tasks="$2"

    for n in `seq 0 $(( njobs - 1 ))`; do

        start=$(( n * tasks ))

        unset cmdtarray
        cmdtarray=( "${cmdarray[@]:start:tasks}" )  # slice array
        # TODO: set tasks to actual number of commands for array jobs

        cmd=$( IFS=$'\n'; echo "${cmdtarray[*]}" )

        jids+=$( single_job "$cmd" )

    done

}


function generate_job_script {
    # Create a script for a single job.
    # The following components are written to the scriptfile:
        # Shebang for bash
        # Any directives specified via $additions
        # The command in $cmd
        # The 'wait' command if $additions specifies it as an array job.
    # Echoes the script path.

    local cmd="$*"
    local scriptfile

    scriptfile=$( generate_job_script_name )

    if [ "$compute_env" != "JAL" ]
    then
        echo '#!/bin/bash' > $scriptfile
        echo "" >> $scriptfile
        echo "$( get_directives )" >> $scriptfile
        echo "" >> $scriptfile
        echo "$cmd" >> $scriptfile
        if [[ $additions == *"array"* ]]
        then
            echo "wait" >> $scriptfile
        fi
    else
        echo "$cmd" > $scriptfile
    fi

    echo "$scriptfile"

}


function generate_job_script_name {
    # Create a filepath for the script.
    # Echoes the script path.

    ext='.sh'

    if [ "$compute_env" == "ARCB" ]
    then
        prefix='EM_sb_'
    elif [ "$compute_env" == "ARC" ]
    then
        prefix='EM_qs_'
    elif [ "$compute_env" == "JAL" ]
    then
        prefix='EM_jp_'
        if [[ $additions == *"matlab"* ]]
        then
            ext='.m'
        fi
    else
        prefix='EM_lc_'
    fi

    postfix=_`printf %03d $n`  # TODO: when njobs is unset

    scriptfile=$datadir/$prefix$jobname$postfix$ext

    echo "$scriptfile"

}


function generate_job_script_partial {

    local scriptfile=$1
    local postfix=$2
    shift 2
    local datastem_indices="$*"

    local ext="${scriptfile##*.}"
    local filename="${scriptfile%.*}"
    local scriptfile_new=$filename$postfix.$ext

    > $scriptfile_new
    for linenumber in $datastem_indices; do
        sed "${linenumber}q;d" $scriptfile >> $scriptfile_new
    done

    echo "$scriptfile_new"

}


function submit_job {
    # Submit a job to a compute cluster.
        # (1) Holds the job when $q='h'
        # (2) Sends to devel queue when $q='d'
        # (3) Sends to compute queue otherwise
    # Echoes either the scriptpath (1) or the sbatch/qsub answer (2-3)

    local subfile="$1"
    local JOB subcmd devq array

    if [ "$compute_env" == "ARCB" ]
    then
        subcmd='sbatch'
        devq='-p devel'
        array=''  # TODO: implement array='--array=100-200'
    elif [ "$compute_env" == "ARC" ]
    then
        subcmd='qsub'
        devq='-q develq'
        array=''  # TODO: implement array='-t 100-200'
    elif [ "$compute_env" == "JAL" ]
    then
        subcmd='fsl_sub -q long.q'
        devq='-q veryshort.q'
        array='-t'
    fi

    if [ "$q" = 'h' ]  # Hold the job without submitting
    then
        subcmd="$subfile"
        JOB=( "$subcmd" )
    else
        if [ "$q" = 'd' ]  # Submit to the development queue
        then
            subcmd+=" $devq"
        fi
        if [[ $additions == *"array"* ]]  # Submit as an array job
        then
            subcmd+=" $array"
        fi
        subcmd+=" $subfile"
        JOB=$( $subcmd )
    fi

    echo "$JOB"

}


function get_command_array_datastems {
    # Generate an array of commands from a block identifier array.
        #

    local cmdfun=$1
    local datastem

    cmdarray=()

    for datastem in "${datastems[@]}"; do

        [ -z "$datastem" ] && continue

        cmdt=$( $cmdfun )

        # TODO: use other term, because these are not really SLURM array jobs
        if [[ $additions == *"array"* ]]  && [[ "$compute_env" == *"ARC"* ]]
        then
            cmdarray+=( "$cmdt &" )
        else
            cmdarray+=( "$cmdt" )
        fi

    done

}


function get_command_array_dataslices {
    #

    local start=$1
    local stop=$2
    local step=$3
    local cmdfun=$4
    local z Z

    cmdarray=()

    for z in `seq $start $step $(( stop - 1 ))`; do

        Z=$( get_coords_upper $z $start $step $stop )

        cmdt=$( $cmdfun )

        if [[ $additions == *"array"* ]]  && [[ "$compute_env" == *"ARC"* ]]
        then
            cmdarray+=( "$cmdt &" )
        else
            cmdarray+=( "$cmdt" )
        fi

    done

}


function get_directives {
    # Get directives, parameters, paths, etc.
        # SBATCH directives (if on SLURM).
        # MPI enabling script (if on Arcus-B).
        # Conda path and environment activation.
        # Ilastik ENV variables.
        # Neuroproof library paths.
        # Definition of PYTHONPATH.
    # Echoes the set of directives.

    # TODO: execute directives on Jalapeno before submitting jobs

    unset directive_funs
    declare -a directive_funs

    if [ "$compute_env" == "ARCB" ]
    then
        directive_funs+=( sbatch_directives )
        directive_funs+=( mpi_directives )
    elif [ "$compute_env" == "ARC" ]
    then
        directive_funs+=( pbs_directives )
        directive_funs+=( mpi_directives )
    fi
    directive_funs+=( conda_directives )
    directive_funs+=( ilastik_directives )
    directive_funs+=( neuroproof_directives )
    directive_funs+=( ppath_directives )

    # only echoes for non-empty directives
    for fun in ${directive_funs[@]}; do
        dir="$( $fun )" &&
            [ -n "$dir" ] &&
                echo "$dir"
    done

}


function pbs_directives {
    # Generate pbs directives for submission script.

    echo "#PBS -l nodes=$nodes:ppn=$tasks"
    echo "#PBS -l mem=$memcpu"
    [ "$q" = "d" ] &&
        echo "#PBS -l walltime=00:10:00" ||
            echo "#PBS -l walltime=$wtime"
    echo "#PBS -N EM_$jobname"
    echo "#PBS -V"
    echo "cd \$PBS_O_WORKDIR"

}


function sbatch_directives {
    # Generate sbatch directives for submission script.

    echo "#SBATCH --nodes=$nodes"
    echo "#SBATCH --ntasks-per-node=$tasks"
    echo "#SBATCH --mem=$memcpu"
    [ "$q" = "d" ] &&
        echo "#SBATCH --time=00:10:00" ||
            echo "#SBATCH --time=$wtime"
    echo "#SBATCH --job-name=EM_$jobname"

}


function mpi_directives {
    # Generate mpi directives for submission script.

    if [[ $additions == *"mpi"* ]]
    then
        if [ "$compute_env" == "ARCB" ]
        then
            if [ "$CONDA_ENV" == "parallel" ]
            then
                echo ". $HOME/workspace/enable_arcus-b_mpi.sh"
            else
                echo ". enable_arcus-b_mpi.sh"
            fi
        elif [ "$compute_env" == "ARC" ]
        then
            echo ". enable_arcus_mpi.sh"
        fi
    fi

}


function conda_directives {
    # Generate conda directives for submission script.

    if [[ $additions == *"conda"* ]]
    then
      echo "export PATH=$( conda info --root ):\$PATH"
      echo "source activate $CONDA_ENV"
    fi

}


function ilastik_directives {
    # Generate ilastik directives for submission script.

    if [[ $additions == *"ilastik"* ]]
    then
        echo "export LAZYFLOW_THREADS=$tasks"
        echo "export LAZYFLOW_TOTAL_RAM_MB=$memcpu"
    fi

}


function neuroproof_directives {
    # Generate neuroproof directives for submission script.

    if [[ $additions == *"neuroproof"* ]]
    then
        echo "export PATH=$CONDA_PATH:\$PATH"
        echo "export LD_LIBRARY_PATH=$CONDA_PATH/envs/$CONDA_ENV/lib"
    fi

}


function ppath_directives {
    # Generate pythonpath directives for submission script.

    if [[ $additions == *"ppath"* ]]
    then
        echo "export PYTHONPATH=$PYTHONPATH"
    fi

}


function cmd_prefix_mpi {
    # Prefix the MPI command.
        # The command to execute with MPI is read from the arguments.
        # Switches between Arcus-B (mpirun) and LOCAL (mpiexec).
    # Echoes the adapted command string.

    local cmd="$*"

    if [[ "$additions" == *"mpi"* ]]
    then
        if [[ "$compute_env" == *"ARC"* ]]
        then
            cmd="mpirun \$MPI_HOSTS ${cmd}"
        elif [ "$compute_env" == "LOCAL" ]
        then
            cmd="mpiexec -n $tasks ${cmd}"
        elif [ "$compute_env" == "RIOS013" ]
        then
            cmd="mpiexec -n $tasks ${cmd}"
        fi
    fi

    echo "$cmd"

}


# function cmd_replace_datastem {
#     # Replace all occurences of 'datastem' in a command string.
#         # Replaced with the block identifier (the first argument).
#         # The command string is read from the remaining arguments.
#     # Echoes the adapted command string.
#
#     local datastem=$1
#     shift
#     local cmd="$*"
#
#     echo "${cmd//datastem/$datastem}"
#
# }


# function get_command {
#     # add the command to execute
#     cmd="$( cmd_prefix_mpi "$cmd" )"
#     cmd="$( cmd_replace_datastem "${datastems[n]}" "$cmd" )"
# }


function cmd_deletelabels {  # TODO
    #

    if [[ $additions == *"deletelabels"* ]]
    then
        deletelabels=`grep $datastem $deletefile | awk '{$1 = ""; print $0;}'`
        cmdt=${cmdt//deletelabels/$deletelabels}
    fi
}


function cmd_mergelabels {  # TODO
    #

    if [[ $additions == *"mergelabels"* ]]
    then
        mergelabels=`grep $datastem $mergefile | awk '{$1 = ""; print $0;}'`
        cmdt=${cmdt//mergelabels/$mergelabels}
    fi
}


function set_vars {
    # Declare a set of variables.
        # Read from the string in $vars.
        # Delimited by semicolon.

    local vars="$*"
    local awkfun='{ for(i=0;i<=NF;i++) {printf "%s\n", $i} }'

    eval $( echo "$vars" | awk -F ';' "$awkfun" )

}


function dm3convert {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    jobname='dm3convert'
    additions='mpi-conda-ppath'
    CONDA_ENV='root'

    cmd=$( get_cmd_dm3convert )

    if [[ "$compute_env" == *"ARC"* ]]; then

        module load mpich2/1.5.3__gcc

        nodes=4
        memcpu=2000
        wtime='01:00:00'

        tasks=16
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=6

    fi

    single_job "$cmd"

}


function fiji_register {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    mkdir -p $datadir/$regname/trans
    adapt_fiji_register

    jobname='fiji_register'
    additions=''
    CONDA_ENV=''

    cmd=$( get_cmd_fiji_register )

    if [[ "$compute_env" == *"ARC"* ]]; then

        nodes=1
        memcpu=60000
        wtime='10:00:00'

        tasks=1
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        :

    fi

    single_job "$cmd"

}


function tif2h5 {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    local opf=$2
    local ods=$3

    jobname='tif2h5'
    additions=''
    CONDA_ENV=''

    cmd=$( get_cmd_series2stack )

    if [[ "$compute_env" == *"ARC"* ]]; then

        module load mpich2/1.5.3__gcc

        nodes=1
        memcpu=30000
        wtime='03:00:00'

        tasks=1
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        :

    fi

    single_job "$cmd"

}


function trainingdata {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    local datastem=$( get_datastem $dataset 3000 3500 3000 3500 0 $zmax )
    local vol_slice=
    local ipf=
    local ids='data'
    local opf=
    local ods='data';

    jobname='trainingdata'
    additions=''
    CONDA_ENV=''

    cmd=$( get_cmd_splitblocks_datastem )

    if [[ "$compute_env" == *"ARC"* ]]; then

        nodes=1
        memcpu=6000
        wtime='10:00:00'

        tasks=1
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        :

    fi

    single_job "$cmd"

}


function apply_ilastik {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    local ipf=$2
    local ids=$3
    local opf=$4
    local ods=$5

    local pixprob_trainingset=$6

    jobname='apply_ilastik'
    additions='ilastik'
    CONDA_ENV=''

    cmd=$( get_cmd_apply_ilastik )

    if [[ "$compute_env" == *"ARC"* ]]; then

        nodes=1
        memcpu=125000
        wtime='99:00:00'

        tasks=16
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=8
        memcpu=15000

    fi

    single_job "$cmd"

}


function split_blocks {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun cmd n

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="splitblocks_bs$bs"
    additions='conda'
    CONDA_ENV='root'

    if [[ "$compute_env" == *"ARC"* ]]; then

        memcpu=6000
        nodes=1

        # CONDA_ENV='parallel'
        # wtime='00:10:00'
        # tasks=4
        # njobs=$(( (n + tasks - 1) / tasks ))

        wtime='05:10:00'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    fi

    fun=get_cmd_splitblocks_datastem
    get_command_array_datastems $fun

    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}


function splitblocks {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun cmd n

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6
    shift 6

    local args=$*

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="splitblocks_bs$bs"
    additions='conda'
    CONDA_ENV='parallel'

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+='-mpi'
        memcpu=6000
        nodes=4
        wtime='01:10:00'
        tasks=16
        njobs=1
        local args+=' -M'

    elif [ "$compute_env" == "JAL" ]; then

        :

    elif [ "$compute_env" == "LOCAL" ]; then

        :

    fi

    cmd=$( get_cmd_splitblocks_mpi )
    single_job "$cmd"

}


function sum_volumes {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun cmd n

    local q=$1
    local stemsmode=$2
    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6
    shift 6
    local vols="$@"

    # local ipf='_probs'
    # local ids='volume/predictions'
    # local opf='_probs'
    # local volsns="$( echo -e "${vols}" | tr -d '[:space:]' )"
    # local ods="sum$volsns"

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="sum_volumes_$ods"
    additions='conda'
    CONDA_ENV='root'

    if [[ "$compute_env" == *"ARC"* ]]; then

        module load hdf5-parallel/1.8.17_mvapich2_gcc

        additions+='-array'

        nodes=1
        memcpu=6000
        wtime='00:10:00'

        tasks=16
        njobs=$(( (n + tasks - 1) / tasks ))

    elif [ "$compute_env" == "JAL" ]; then

        additions+='-array'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    fi

    fun=get_cmd_sum_volumes
    get_command_array_datastems $fun

    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}


function eed {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local nstems

    local q=$1

    local stemsmode=$2  # m: missing, a: all, unset/empty: use ENV

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6

    local wtime=$7

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="${FUNCNAME[0]}$ipf.${ids////-}"
    additions=''
    CONDA_ENV=''

    if [[ "$compute_env" == *"ARC"* ]]
    then

        additions+='-array'

        local fun=get_cmd_eed_deployed
        get_command_array_datastems $fun

        # module load hdf5-parallel/1.8.17_mvapich2_gcc
        # module load matlab/R2015a

        nodes=1
        memcpu=50000

        tasks=16
        njobs=$(( (n + tasks - 1) / tasks ))

        unset JOBS && declare -a JOBS
        array_job $njobs $tasks

    elif [ "$compute_env" == "JAL" ]
    then

        additions+='-array'
        additions+='-matlab'

        local fun=get_cmd_eed_matlab
        get_command_array_datastems $fun

        tasks=$n
        njobs=1

        cmd=$( IFS=$'\n'; echo "${cmdarray[*]}" )
        single_job "$cmd"

    elif [ "$compute_env" == "LOCAL" ]
    then

        # local fun=get_cmd_eed_matlab  # TODO
        local fun=get_cmd_eed_deployed
        get_command_array_datastems $fun

        tasks=$n
        njobs=1

        unset JOBS && declare -a JOBS
        array_job $njobs $tasks

    fi

}


function mergeblocks {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime

    local q=$1

    local Vstart=$2
    local Vtasks=$3

    local ipf=$4
    local ids=$5
    local opf=$6
    local ods=$7

    shift 7

    local args="$*"

    jobname="${FUNCNAME[0]}$ipf.${ids////-}"
    additions=''
    CONDA_ENV='parallel'  # 'root'

    if [ "$compute_env" == "ARCB" ]; then
        additions+='-conda-mpi'
        nodes=4
        memcpu=30000
        wtime='03:10:00'
        tasks=16
        njobs=1
        # args='-B 1 7 7 -f np.mean'
    elif [ "$compute_env" == "ARC" ]; then
        additions+='-mpi'
        nodes=1
        memcpu=30G  # TODO: for all submission functions
        wtime='03:10:00'
        tasks=16
        njobs=1
    elif [ "$compute_env" == "JAL" ]; then
        :  # TODO
    elif [ "$compute_env" == "LOCAL" ]; then
        :
    elif [ "$compute_env" == "RIOS013" ]; then
        :
    fi

    # TODO separate jobs for infiles array of selections...?
    if [ "$Vtasks" != "0" ]
    then
        unset infiles
        declare -a infiles
        get_infiles_datastems

        [ ! -z "$Vstart" ] && [ ! -z "$Vtasks" ] &&
            infiles=( "${infiles[@]:Vstart:Vtasks}" )  # slice array
    fi

    cmd=$( get_cmd_mergeblocks )

    single_job "$cmd"

}


function prob2mask {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime

    local q=$1

    local ipf=$2
    local ids=$3
    local opf=$4
    local ods=$5

    local blocksize=$6

    shift 6
    local args="$*"

    jobname="p2m_$ods"
    if [ "$compute_env" == "ARCB" ]; then
        additions='conda'
    elif [ "$compute_env" == "ARC" ]; then
        additions='array'
    fi
    CONDA_ENV='root'

    nblocks=$(( (zmax + blocksize - 1) / blocksize ))

    local fun=get_cmd_prob2mask
    get_command_array_dataslices 0 $zmax $blocksize $fun

    if [[ "$compute_env" == *"ARC"* ]]; then

        nodes=1
        memcpu=30000
        wtime='01:10:00'
        tasks=1
        njobs=1

        cmd=$( IFS=$'\n'; echo "${cmdarray[*]}" )
        single_job "$cmd"

    elif [ "$compute_env" == "JAL" ]; then

        cmd=$( IFS=$'\n'; echo "${cmdarray[*]}" )
        single_job "$cmd"

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$nblocks
        njobs=1
        unset JOBS && declare -a JOBS
        array_job $njobs $tasks

    fi

    # unset JOBS && declare -a JOBS
#     array_job $njobs $tasks

}


function prob2mask_datastems {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun cmd n

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6

    shift 6
    local arg="$@"

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="p2m_dstems_$ods"

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions='array'

        nodes=1
        memcpu=30000
        wtime='01:10:00'
        tasks=16
        njobs=$(( (n + tasks - 1) / tasks ))

        fun=get_cmd_prob2mask_datastems
        get_command_array_datastems $fun
        unset JOBS && declare -a JOBS
        array_job $njobs $tasks

    elif [ "$compute_env" == "JAL" ]; then

	    additions='conda'
	    CONDA_ENV='root'
        fun=get_cmd_prob2mask_datastems
        get_command_array_datastems $fun
        cmd=$( IFS=$'\n'; echo "${cmdarray[*]}" )
        single_job "$cmd"

    elif [ "$compute_env" == "LOCAL" ]; then

        additions='conda'
        CONDA_ENV='root'
        tasks=$n
        njobs=1

        fun=get_cmd_prob2mask_datastems
        get_command_array_datastems $fun
        unset JOBS && declare -a JOBS
        array_job $njobs $tasks

    fi

}


function blockreduce {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime

    local q=$1

    local ipf=$2
    local ids=$3
    local opf=$4
    local ods=$5

    local brfun=$6
    local brvol=$7
    local blocksize=$8
    local memcpu=$9
    local wtime=${10}
    shift 10
    local vol_slice="$*"

    jobname="rb$ipf.${ids////-}"  # remove slashes
    additions='conda'
    CONDA_ENV='root'

    nblocks=$(( (zmax + blocksize - 1) / blocksize ))

    if [[ "$brfun" == "expand" ]]; then
        local fun=get_cmd_downsample_blockwise_expand
    else
        local fun=get_cmd_downsample_blockwise
    fi

    get_command_array_dataslices 0 $zmax $blocksize $fun

    if [[ "$compute_env" == *"ARC"* ]]; then

        nodes=1
        tasks=1
        njobs=1

        cmd=$( IFS=$'\n'; echo "${cmdarray[*]}" )
        single_job "$cmd"

    elif [ "$compute_env" == "JAL" ]; then

        cmd=$( IFS=$'\n'; echo "${cmdarray[*]}" )
        single_job "$cmd"

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$nblocks
        njobs=1
        unset JOBS && declare -a JOBS
        array_job $njobs $tasks

    fi

}


function h52nii {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1
    local dataroot=$2
    local ipf=$3
    local ids=$4
    shift 6
    local args="$@"

    local opf=$5
    local ods=$6

    [ -z "$opf" ] && opf=$ipf
    [ -z "$ods" ] && ods=${ids////-}

    jobname="${FUNCNAME[0]}$ipf.${ids////-}"  # remove slashes
    additions='conda'
    CONDA_ENV='root'

    cmd=$( get_cmd_h52nii $dataroot )

    if [[ "$compute_env" == *"ARC"* ]]; then

        nodes=1
        memcpu=6000
        wtime='00:10:00'

        tasks=1
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        :

        elif [ "$compute_env" == "LOCAL" ]; then

        :

    fi

    single_job "$cmd"

}


function conncomp {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    local mode=$2  # 2D 2Dfilter 2Dprops 2Dto3D (3D) train test
    local dataroot=$3

    local ipf=$4
    local ids=$5
    local opf=$6
    local ods=$7

    local MEANINT="--maskMB $datadir/$dataroot$8"

    local clfpath=$9
    local scalerpath=${10}

    shift 10
    local args=$*

    props=( 'label' 'area' 'eccentricity' 'mean_intensity' \
        'solidity' 'extent' 'euler_number' )

    jobname="${FUNCNAME[0]}_$dataroot$ipf.${ids////-}.$mode"  # remove slashes
    additions='conda'
    CONDA_ENV='parallel'

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+='-mpi'
        mpiflag='-M'

        nodes=4
        memcpu=60000
        wtime='05:10:00'
        tasks=16
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        mpiflag=''

    elif [ "$compute_env" == "LOCAL" ]; then

        additions+='-mpi'  # FIXME: no mpi for 2Dprops
        mpiflag='-M'
        tasks=7  # 7 props

        source activate scikit-image-devel_0.13

    fi

    cmd=$( get_cmd_${FUNCNAME[0]}_$mode $dataroot )

    single_job "$cmd"

}


function conncomp_3D {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    local dataroot=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6

    shift 6
    local args="$*"

    jobname="cc3D"
    additions=''
    CONDA_ENV=''

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+=''
        mpiflag=''

        nodes=1
        memcpu=60000
        wtime='05:10:00'
        tasks=1
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        mpiflag=''

    elif [ "$compute_env" == "LOCAL" ]; then

        additions+=''
        mpiflag=''

        source activate scikit-image-devel_0.13

    fi

    cmd=$( get_cmd_${FUNCNAME[0]} $dataroot "$args")

    single_job "$cmd"

}


# function simple {
#     local q=$1
#     local dataroot=$2
#     local ipf=$3
#     local ids=$4
#     local opf=$5
#     local ods=$6
#     shift 6
#     local args="$*"
#     jobname="NoR"
#     additions=''
#     CONDA_ENV=''
#     cmd=$( get_cmd_${FUNCNAME[0]} $dataroot "$args")
#     single_job "$cmd"
# }
# # NoR=$(declare -f simple)  # remap combine_labels


function NoR {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    local dataroot=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6

    shift 6
    local args="$*"

    jobname="NoR"
    additions=''
    CONDA_ENV=''

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+=''
        mpiflag=''

        nodes=1
        memcpu=60000
        wtime='05:10:00'
        tasks=1
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        mpiflag=''

    elif [ "$compute_env" == "LOCAL" ]; then

        additions+=''
        mpiflag=''

        source activate scikit-image-devel_0.13

    fi

    cmd=$( get_cmd_${FUNCNAME[0]} $dataroot "$args")

    single_job "$cmd"

}


function remap {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    local dataroot=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6

    shift 6
    local args="$*"

    jobname="remap"
    additions=''
    CONDA_ENV=''

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+=''
        mpiflag=''

        nodes=1
        memcpu=60000
        wtime='05:10:00'
        tasks=1
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        mpiflag=''

    elif [ "$compute_env" == "LOCAL" ]; then

        additions+=''
        mpiflag=''

        source activate scikit-image-devel_0.13

    fi

    cmd=$( get_cmd_${FUNCNAME[0]} $dataroot "$args")

    single_job "$cmd"

}


function combine_labels {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1
    local dataroot=$2
    shift 2
    local args="$*"

    jobname="combine_labels"
    additions=''
    CONDA_ENV=''

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+=''
        mpiflag=''

        nodes=1
        memcpu=60000
        wtime='05:10:00'
        tasks=1
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        mpiflag=''

    elif [ "$compute_env" == "LOCAL" ]; then

        additions+=''
        mpiflag=''

        source activate scikit-image-devel_0.13

    fi

    cmd=$( get_cmd_${FUNCNAME[0]} $dataroot "$args")

    single_job "$cmd"

}


function merge_slicelabels {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1
    local dataroot=$2
    shift 2
    local args="$*"

    jobname="combine_labels"
    additions=''
    CONDA_ENV=''

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+=''
        mpiflag=''

        nodes=1
        memcpu=60000
        wtime='05:10:00'
        tasks=1
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        mpiflag=''

    elif [ "$compute_env" == "LOCAL" ]; then

        additions+=''
        mpiflag=''

        source activate scikit-image-devel_0.13

    fi

    cmd=$( get_cmd_${FUNCNAME[0]} $dataroot "$args")

    single_job "$cmd"

}


function merge_slicelabels_mpi {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1
    local dataroot=$2
    shift 2
    local args="$*"

    jobname="merge_slicelabels"
    additions='conda'
    CONDA_ENV='parallel'

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+='-mpi'
        mpiflag='-M'

        nodes=4
        memcpu=60000
        wtime='05:10:00'
        tasks=16
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        mpiflag=''

    elif [ "$compute_env" == "LOCAL" ]; then

        additions+='-mpi'
        mpiflag='-M'
        tasks=7  # 7 props

        source activate scikit-image-devel_0.13

    elif [ "$compute_env" == "RIOS013" ]; then

        additions+='-mpi'
        mpiflag='-M'
        tasks=14

        CONDA_ENV='h5para'
        source activate h5para

    fi

    cmd=$( get_cmd_${FUNCNAME[0]} $dataroot "$args")

    single_job "$cmd"

}


function slicvoxels {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun nstems

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6

    local l=$7
    local c=$8
    local s=$9

    local tasks=${10}

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="slic_$ods"
    additions='conda'
    CONDA_ENV='root'

    local fun=get_cmd_slicvoxels
    get_command_array_datastems $fun

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+='-array'

        nodes=1
        # memcpu=60000; wtime='01:10:00';
        memcpu=125000; wtime='05:10:00';

        # tasks=6  # 8 GB per process for 184x500x500
        njobs=$(( (n + tasks - 1) / tasks ))

    elif [ "$compute_env" == "JAL" ]; then

        additions+='-array'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    fi

    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}


function smoothdata {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun nstems

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6
    shift 6
    local args=$*

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="smooth_$ods"
    additions='conda'
    CONDA_ENV='root'

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+='-array'

        nodes=1
        memcpu=125000; wtime='00:30:00';

        tasks=16
        njobs=$(( (n + tasks - 1) / tasks ))

    elif [ "$compute_env" == "JAL" ]; then

        additions+='-array'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    fi

    local fun=get_cmd_smooth
    get_command_array_datastems $fun
    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}


function watershed_ics {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun nstems

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6
    local mpf=$7
    local mds=$8
    shift 8
    local args=$*

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="ws_$ods"

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions='array'

        nodes=1
        # memcpu=125000;
        wtime='03:10:00';
        tasks=16
        njobs=$(( (n + tasks - 1) / tasks ))

    elif [ "$compute_env" == "JAL" ]; then

        additions='array'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    fi

    local fun=get_cmd_watershed_ics
    get_command_array_datastems $fun
    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}


function agglo_mask {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun nstems

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6
    local lpf=$7
    local lds=$8
    shift 8
    local args=$*

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="ws_$ods"

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions='array'

        nodes=1
        # memcpu=125000;
        wtime='03:10:00';
        tasks=16
        njobs=$(( (n + tasks - 1) / tasks ))

    elif [ "$compute_env" == "JAL" ]; then

        additions='array'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    fi

    local fun=get_cmd_agglo_mask
    get_command_array_datastems $fun
    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}


function upsample_blocks {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun nstems

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6
    shift 6
    local args=$*

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="us_$ods"

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions='array'

        nodes=1
        # memcpu=125000;
        wtime='03:10:00';
        tasks=16
        njobs=$(( (n + tasks - 1) / tasks ))

    elif [ "$compute_env" == "JAL" ]; then

        additions='array'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    fi

    local fun=get_cmd_upsample_blocks
    get_command_array_datastems $fun
    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}


function fill_holes {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun nstems

    local q=$1

    local stemsmode=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6
    shift 6
    local args=$*

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="fh_$ods"

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions='array'

        nodes=1
        # memcpu=125000;
        wtime='03:10:00';
        tasks=16
        njobs=$(( (n + tasks - 1) / tasks ))

    elif [ "$compute_env" == "JAL" ]; then

        additions='array'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    elif [ "$compute_env" == "RIOS013" ]; then

        tasks=20
        njobs=$(( (n + tasks - 1) / tasks ))

    fi

    local fun=get_cmd_fill_holes
    get_command_array_datastems $fun
    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}


function merge_labels_ws {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local cmd

    local q=$1

    local dataroot=$2

    local ipf=$3
    local ids=$4
    local opf=$5
    local ods=$6

    shift 6
    local args=$*

    jobname="ws"
    additions='conda'
    CONDA_ENV='parallel'

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions+='-mpi'
        mpiflag='-M'

        nodes=4
        memcpu=60000
        wtime='05:10:00'
        tasks=16
        njobs=1

    elif [ "$compute_env" == "JAL" ]; then

        mpiflag=''

    elif [ "$compute_env" == "LOCAL" ]; then

        additions+='-mpi'
        mpiflag='-M'
        tasks=7  # 7 props

        source activate scikit-image-devel_0.13

    elif [ "$compute_env" == "RIOS013" ]; then

        # FIXME: suspect bad mpi-implementation of this function
        # additions+='-mpi'
        # mpiflag='-M'
        # tasks=7  # 7 props
        # CONDA_ENV='h5para'
        additions+=''
        mpiflag=''
        # tasks=7  # 7 props
        CONDA_ENV=''

    fi

    cmd=$( get_cmd_${FUNCNAME[0]} $dataroot )

    single_job "$cmd"

}


function separate_sheaths {
    #

    local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
    local fun nstems

    local q=$1

    local stemsmode=$2
    local fun=$3

    local ipf=$4
    local ids=$5
    local opf=$6
    local ods=$7
    shift 7
    local args=$*

    set_datastems $stemsmode
    n=${#datastems[@]}

    jobname="fh_$ods"

    if [[ "$compute_env" == *"ARC"* ]]; then

        additions='array'

        nodes=1
        mem=8000;
        wtime='01:10:00';
        tasks=8
        njobs=$(( (n + tasks - 1) / tasks ))

    elif [ "$compute_env" == "JAL" ]; then

        additions='array'
        tasks=$n
        njobs=1

    elif [ "$compute_env" == "LOCAL" ]; then

        tasks=$n
        njobs=1

    elif [ "$compute_env" == "RIOS013" ]; then

        tasks=20
        njobs=$(( (n + tasks - 1) / tasks ))

    fi

    get_command_array_datastems $fun
    unset JOBS && declare -a JOBS
    array_job $njobs $tasks

}

