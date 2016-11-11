# sudo ln -s ~/anaconda/envs/scikit-image-devel_0.13 /opt/anaconda1anaconda2anaconda3
# source activate scikit-image-devel_0.13
# conda install python=2.7
# conda install -c spectralDNS hdf5-parallel=1.8.14
# conda install -c spectralDNS h5py-parallel=2.6.0
# conda install --channel mpi4py mpich mpi4py
# pip install git+git://github.com/scikit-image/scikit-image@master
# pip install nibabel

source activate scikit-image-devel_0.13
scriptdir="${HOME}/workspace/EM"
datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/test"
# dset_name="M3S1GNU_06950-08050_05950-07050_00030-00460"
dset_name="M3S1GNU_06950-08050_04950-06050_00030-00460"
datastem=$dset_name

python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '3D' \
--maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack' \
-o '_labelMA_core3D' -a 10000

# .1GB
mpirun -np 8 python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2D' -o '_labelMA_core2D' -m

# .1GB
mpirun -np 8 python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dfilter' -m \
-i '_labelMA_core2D' -o '_labelMA_core2D_fw' \
-a 200 -A 30000 -e 0 -x 0.30 -s 0.50 \
-p 'label' 'area' 'mean_intensity' 'eccentricity' 'solidity' 'extent' 'euler_number'

# 4GB then 2GB
mpirun -np 2 python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dprops' \
-i '_labelMA_core2D' -o '_labelMA_core2D_fw' \
-p 'label' 'area' 'mean_intensity' 'eccentricity' 'solidity' 'extent' 'euler_number'
# area might be very useful (set to between 200 and 30000)
# mean_intensity does not seem to be very useful
# eccentricity does not seem to be very useful
# solidity might be moderately useful (set to 0.5)
# euler_number might be very useful (set to 0 or 1)
# extent might be moderately useful (set to 0.3)
# TODO: evaluate these features:
# inertia_tensor_eigvals
# major_axis_length
# minor_axis_length
# moments
# moments_central
# moments_hu
# perimeter  #processed

# 8GB
python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dto3Dlabel' \
-o '_labelMA_core2D_fw'

# 10GB
python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir $datastem \
-l '_labelMA_core2D_fw_3Dlabeled' 'stack' \
-m 4 -t 0.50 -s 10000 -o '_labelMA_core2D_fw_3Dmerged'





for propname in 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number' '3Dlabeled'; do
    pf="_labelMA_core2D_fw_${propname}"
    python $scriptdir/convert/EM_stack2stack.py \
    $datadir/${datastem}${pf}.h5 $datadir/${datastem}${pf}.nii.gz \
    -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' &
done




# core2D without saving maxlabel can be 'corrected':
    if mode == 'TMP':
        fname = os.path.join(datadir, dset_name + inpf + '.h5')
        f = h5py.File(fname, 'r')
        n_slices = f['stack'].shape[0]
        local_nrs = np.array(range(0, n_slices), dtype=int)
        maxlabel = 0
        for i in local_nrs:
            maxlabel = max(maxlabel, np.amax(f['stack'][i,:,:]))

        filename = os.path.join(datadir, dset_name + inpf + '.npy')
        np.save(filename, np.array([maxlabel]))
        print(maxlabel)

        f.close()



python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M 'TMP' -i '_labelMA_core2D'

# for f in `ls *core2D_constraints*`; do
#     mv ${f} ${f/core2D_constraints/core2D}
# done


unset datastems
declare -a datastems
datastems[0]=$dataset
export datastems
export jobname="cc2Dfilter"
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=3000 wtime="10:00:00" q=""
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dfilter' \
-i '_labelMA_core2D' -o '_labelMA_core2D_fw_criteria' -E 1 \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh


source datastems_09blocks.sh
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=${#datastems[@]} nodes=1 tasks=1 memcpu=125000 wtime="10:00:00" q=""
export jobname="prop_9b"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem \
-l '_labelMA_core2D_labeled' 'stack' --maskMM '_maskMM' 'stack' \
-m 4 -t 0.01 -s 10000 -o '_labelMA_core2D_merged'"
source $scriptdir/pipelines/template_job_$template.sh
