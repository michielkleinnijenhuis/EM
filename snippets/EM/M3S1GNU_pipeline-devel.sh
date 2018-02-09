ssh -Y jalapeno.fmrib.ox.ac.uk
ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/M3S1GNU"
localdir="/vols/Data/km/michielk/"
rsync -avz $remdir/${f} $localdir
rsync -avz $localdir/${f} $remdir

###=========================================================================###
### environment prep
###=========================================================================###

export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/M3S1GNU" && mkdir -p $datadir && cd $datadir
# source datastems_90blocks.sh

###=========================================================================###
### stitch (not performed again)
###=========================================================================###

###=========================================================================###
### register (performed again on previously stitched - ref to 0250.tif)
###=========================================================================###

###=========================================================================###
### downsample
###=========================================================================###

###=========================================================================###
### Create the stack (fails with mpi4py, but there's no need)
###=========================================================================###

cp ../EM/M3/archive/M3_S1_GNU_old/M3S1GNU/M3S1GNU.h5 .
cp ../EM/M3/archive/M3_S1_GNU_old/M3S1GNUds7/M3S1GNUds7.h5 .
cp ../EM/M3/archive/M3_S1_GNU_old/datastems_90blocks.sh .
cp ../EM/M3/archive/M3_S1_GNU_old/pixprob_training_arcus.ilp .
cp ../EM/M3/archive/M3_S1_GNU/pixprob_training.h5 .
rsync -Pavz ~/oxdata/P01/EM/M3/M3_S1_GNU/m000/pixprob_training_relpath.ilp $remdir/

###=========================================================================###
### split volume in blocks
###=========================================================================###

def block_ranges(size, margin, full):
    lower = [0] + range(size - margin, full, size)
    upper = range(size + margin, full, size) + [full]
    return lower, upper

def create_regrefs(roisets, roiname, slices):
    """"""
    ref_dtype = h5py.special_dtype(ref=h5py.RegionReference)
    refs = roisets.create_dataset(roiname, (nblocks,), dtype=ref_dtype)
    for i, roi in enumerate(slices):
        # TODO: create attr with ROI descriptions
        refs[i] = ds.regionref[roi[2], roi[1], roi[0]]  #zyx assumed here
    return refs

# 90 blocks
xmax=9179; ymax=8786; zmax=430;
xs=1000; ys=1000; zs=430;
xm=50; ym=50; zm=0;
# 09 blocks
xmax=1311; ymax=1255; zmax=430;
xs=438; ys=419; zs=430;
xm=0; ym=0; zm=0;

xr, Xr = block_ranges(xs, xm, xmax)
yr, Yr = block_ranges(ys, ym, ymax)
zr, Zr = block_ranges(zs, zm, zmax)

slices = [[slice(x, X), slice(y, Y), slice(z, Z)]
          for z, Z in zip(zr, Zr)
          for y, Y in zip(yr, Yr)
          for x, X in zip(xr, Xr)]

nblocks = len(slices)

## create
import h5py
f = h5py.File('M3S1GNUds7.h5', 'a')
ds = f['stack']
roisets = f.require_group('ROIsets')
roiname = 'blocks'
refs = create_regrefs(roisets, roiname, slices)
f.close()

## test
import h5py
f = h5py.File('M3S1GNUds7.h5', 'a')
ds = f['stack']
refs = f['ROIsets/ROI01']
subset = ds[refs[0]]




for sl in slices:
    print(sl)




###=========================================================================###
### apply Ilastik classifier #
###=========================================================================###

### create dataset /volume/predictions
### create links to blocks of this dataset

# export LAZYFLOW_THREADS=1
# export LAZYFLOW_TOTAL_RAM_MB=12000
# ${CONDA_PATH}/envs/ilastik-devel/run_ilastik.sh --headless \
# --preconvert_stacks \
# --project=$datadir/pixprob_training_relpath.ilp \
# --output_format=hdf5 \
# --output_filename_format=$datadir/M3S1GNUds7_probs.h5 \
# --output_internal_path=volume/predictions \
# --output_axis_order=zyxc \
# $datadir/M3S1GNUds7.h5/stack


# export template='single' additions='conda' CONDA_ENV="ilastik-devel"
# export njobs=1 nodes=1 tasks=16 memcpu=8000 wtime="01:30:00" q=""
# export jobname="ilastik"
# export cmd="${CONDA_PATH}/envs/ilastik-devel/run_ilastik.sh --headless \
# --preconvert_stacks \
# --project=$datadir/pixprob_training_relpath.ilp \
# --output_format=hdf5 \
# --output_filename_format=$datadir/M3S1GNUds7_probs.h5 \
# --output_internal_path=volume/predictions2 \
# --cutout_subregion=\"[(0, 0, 0, None), (50, 100, 200, None)]\" \
# --output_axis_order=zyxc \
# $datadir/M3S1GNUds7.h5/stack"
# source $scriptdir/pipelines/template_job_$template.sh

# --cutout_subregion=\"[(0, 0, 0, None), (50, 100, 200, None)]\" \
# --cutout_subregion=\"[(50, 100, 200, None), (100, 200, 400, None)]\" \


export LAZYFLOW_THREADS=16
export LAZYFLOW_TOTAL_RAM_MB=40000
export template='single' additions='conda' CONDA_ENV="ilastik-devel"
export njobs=1 nodes=1 tasks=16 memcpu=60000 wtime="99:00:00" q=""
export jobname="ilastik-all"
export cmd="${CONDA_PATH}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/pixprob_training_relpath.ilp \
--output_format=hdf5 \
--output_filename_format=$datadir/M3S1GNU_probs.h5 \
--output_internal_path=volume/predictions \
--output_axis_order=zyxc \
$datadir/M3S1GNU.h5/stack"
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### EED
###=========================================================================###
module load hdf5-parallel/1.8.14_mvapich2_intel
module load python/2.7__gcc-4.8
module load matlab/R2015a

mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/snippets/eed/EM_eed.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b
cd $datadir

export template='array' additions=""
export njobs=1 nodes=1 tasks=1 memcpu=50000 wtime="05:00:00" q=""

layer=1
export jobname="EED${layer}_last"
export cmd="$datadir/bin/EM_eed \
'$datadir' 'datastem_probs' '/volume/predictions' '/stack' $layer \
> $datadir/datastem_probs0_eed2.log"
source $scriptdir/pipelines/template_job_$template.sh
layer=2
export jobname="EED${layer}"
export cmd="$datadir/bin/EM_eed \
'$datadir' 'datastem_probs' '/volume/predictions' '/stack' $layer \
> $datadir/datastem_probs1_eed2.log"
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### to zyx(c) - moved originals to probs_backup
###=========================================================================###

###=========================================================================###
### base data block nifti's
###=========================================================================###

###=========================================================================###
### maskDS, maskMM, maskMM-0.02, maskMB
###=========================================================================###

###=========================================================================###
### merge blocks
###=========================================================================###
