#     local Vstart=$2
#     local Vtasks=$3
#     declare -a VARSETS
#     VARSETS[0]="ipf=; ids='data'; \
#         opf='_masks'; ods='maskDS'; \
#         arg='-g -l 0 -u 10000000'; \
#         blocksize=20; vol_slice=;"  # TODO: dilate?
#     VARSETS[1]="ipf='_probs_eed'; ids='sum0247_eed'; \
#         opf='_masks'; ods='maskMM'; \
#         arg='-g -l 0.5 -s 2000 -d 1 -S'; \
#         blocksize=12; vol_slice=;"
#     VARSETS[2]="ipf='_probs_eed_sum16'; ids='sum16_eed'; \
#         opf='_masks'; ods='maskICS'; \
#         arg='-g -l 0.2'; \
#         blocksize=12; vol_slice=;"
#     VARSETS[3]="ipf='_probs1_eed'; ids='probMA'; \
#         opf='_masks'; ods='maskMA'; \
#         arg='-g -l 0.2'; \
#         blocksize=10; vol_slice=;"
# #     VARSETS[3]="ipf='_probs_eed'; ids='probs_eed'; \
# #         opf='_masks'; ods='maskMA'; \
# #         arg='-g -l 0.2'; \
# #         blocksize=10; vol_slice='1 2 1';"  # NOTE: keep blocksize < 10
#     VARSETS[4]="ipf='_masks_maskMM_raw'; ids='maskMM_raw'; \
#         opf='_masks_maskMM'; ods='maskMM'; \
#         arg='-g -l 0 -u 0 -s 2000 -d 1';\
#          blocksize=12; vol_slice=;"
#         # NOTE: filtering after blockwise thresholding with prob2mask_datastems
#         # This circumvents the need for huge datafiles (mergeblocks on masks, not data)
#
#     [ ! -z "$Vstart" ] && [ ! -z "$Vtasks" ] &&
#         VARSETS=( "${VARSETS[@]:Vstart:Vtasks}" )  # slice array
#
#     for vars in "${VARSETS[@]}"; do
#
#         set_vars "$vars"



#     local Vstart=$3
#     local Vtasks=$4
#     unset VARSETS
#     declare -a VARSETS
#     VARSETS[0]="ipf=; ids='data'; \
#         opf='_masks'; ods='maskDS'; \
#         arg='-g -l 0 -u 10000000';"  # TODO: dilate?
#     VARSETS[1]="ipf='_probs_eed'; ids='sum0247_eed'; \
#         opf='_masks'; ods='maskMM'; \
#         arg='-g -l 0.5 -s 2000 -d 1 -S';"
#     VARSETS[2]="ipf='_probs_eed'; ids='sum16_eed'; \
#         opf='_masks'; ods='maskICS'; \
#         arg='-g -l 0.2';"
#     VARSETS[3]="ipf='_probs_eed'; ids='probs_eed'; \
#         opf='_masks'; ods='maskMA'; \
#         arg='-g -l 0.2 -D 0 0 1 0 0 1 0 0 1 1 2 1';"
#     [ ! -z "$Vstart" ] && [ ! -z "$Vtasks" ] &&
#         VARSETS=( "${VARSETS[@]:Vstart:Vtasks}" )
#
#     for vars in "${VARSETS[@]}"; do
#         set_vars "$vars"

# function blockreduce {
#     #
#
#     local jobname additions CONDA_ENV njobs nodes tasks memcpu wtime
#
#     local q=$1
#     local Vstart=$2
#     local Vtasks=$3
#
#     declare -a VARSETS
#     # downsample data with 'np.mean': work with datablocks
#     VARSETS[0]="ipf=; ids=data; \
#         opf=; ods=data; \
#         brfun='np.mean'; vol_br=; \
#         blocksize=20; vol_slice=; \
#         memcpu=60000; wtime=02:00:00; q=;"
#     VARSETS[1]="ipf=_probs_eed_sum0247; ids=sum0247_eed; \
#         opf=_probs_eed_sum0247; ods=sum0247_eed; \
#         brfun='np.mean'; vol_br=; \
#         blocksize=12; vol_slice=; \
#         memcpu=60000; wtime=02:00:00; q=;"
#     VARSETS[2]="ipf=_probs_eed_sum16; ids=sum16_eed; \
#         opf=_probs_eed_sum16; ods=sum16_eed; \
#         brfun='np.mean'; vol_br=; \
#         blocksize=12; vol_slice=; \
#         memcpu=60000; wtime=02:00:00; q=;"
#     VARSETS[3]="ipf=_probs_eed; ids=probs_eed; \
#         opf=_probs_eed; ods=probs_eed; \
#         brfun='np.mean'; vol_br=1; \
#         blocksize=10; vol_slice='0 0 1'; \
#         memcpu=125000; wtime=10:00:00; q=;"
# #     VARSETS[4]="ipf=_probs; ids=volume/predictions; \
# #         opf=_probs; ods=volume/predictions; \
# #         brfun='np.mean'; vol_br=1; \
# #         blocksize=10; vol_slice='0 0 1'; \
# #         memcpu=125000; wtime=10:00:00; q=;"
#     VARSETS[4]="ipf=_probs1; ids=probMA; \
#         opf=_probs1; ods=probMA; \
#         brfun='np.mean'; vol_br=; \
#         blocksize=12; vol_slice=; \
#         memcpu=60000; wtime=02:00:00; q=;"
#     # downsample masks with 'np.amax'; can work with the full volume (on ARC, not JAL)
#     VARSETS[5]="ipf=_masks_maskDS; ids=maskDS; \
#         opf=_masks; ods=maskDS; \
#         brfun='np.amax'; vol_br=; \
#         blocksize=28; vol_slice=; \
#         memcpu=60000; wtime=00:10:00; q='d';"
#     VARSETS[6]="ipf=_masks_maskMM; ids=maskMM;
#         opf=_masks; ods=maskMM; \
#         brfun='np.amax'; vol_br=; \
#         blocksize=27; vol_slice=; \
#         memcpu=60000; wtime=00:10:00; q='d';"
#     VARSETS[7]="ipf=_masks_maskMA; ids=maskMA; \
#         opf=_masks; ods=maskMA; \
#         brfun='np.amax'; vol_br=; \
#         blocksize=27; vol_slice=; \
#         memcpu=60000; wtime=00:10:00; q='d';"
#     VARSETS[8]="ipf=_masks_maskICS; ids=maskICS; \
#         opf=_masks; ods=maskICS; \
#         brfun='np.amax'; vol_br=; \
#         blocksize=27; vol_slice=; \
#         memcpu=60000; wtime=00:10:00; q='d';"
#
#     [ ! -z "$Vstart" ] && [ ! -z "$Vtasks" ] &&
#         VARSETS=( "${VARSETS[@]:Vstart:Vtasks}" )  # slice array
#
#     for vars in "${VARSETS[@]}"; do
#
#         set_vars "$vars"
#
# #         q='h'
#         jobname="rb$ipf.${ids////-}"  # remove slashes
#         additions='conda-ppath'
#         CONDA_ENV='root'
#
#         nblocks=$(( (zmax + blocksize - 1) / blocksize ))
#
#         local fun=get_cmd_downsample_blockwise
#         get_command_array_dataslices 0 $zmax $blocksize $fun
#
#         if [[ "$compute_env" == *"ARC"* ]]; then
#
#             nodes=1
#             tasks=16
#             njobs=$(( (nblocks + tasks - 1) / tasks ))
#
#         elif [ "$compute_env" == "JAL" ]; then
#
#             tasks=$nblocks
#             njobs=1
#
#         elif [ "$compute_env" == "LOCAL" ]; then
#
#             tasks=$nblocks
#             njobs=1
#
#         fi
#
#         array_job $njobs $tasks
#
#     done
# }
