exec '/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia'
alias julia=/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia

Pkg.update()
Pkg.add("Watershed")
Pkg.checkout("Watershed")

### test
using Watershed
aff = rand(Float32, 124,124,12,3);
# watershed(aff)
#println("watershed ...")
#@time watershed(aff)
# @profile watershed(aff)
# Profile.print()
low = 0.1
high = 0.8
thresholds = []
dust_size = 1
watershed(aff; is_threshold_relative=true)

### test assets
chmod +x ~/.julia/v0.6/Watershed/test/test_assets.jl
julia ~/.julia/v0.6/Watershed/test/test_assets.jl

Pkg.add("EMIRT")
Pkg.add("HDF5")

using Watershed
using EMIRT
using HDF5
using Base.Test

aff = h5read(joinpath("/Users/michielk/.julia/v0.6/Watershed", "assets/piriform.aff.h5"), "main")

#seg = atomicseg(aff)
seg, rg = watershed(aff; is_threshold_relative = true)
#h5write(joinpath(Pkg.dir(), "Watershed/assets/piriform.seg.h5", "seg", seg)
#h5write("seg.h5", "seg", seg)
# compare with segmentation

seg0 = readseg(joinpath(("/Users/michielk/.julia/v0.6/Watershed", "assets/piriform.seg.h5"))

err = segerror(seg0, seg)
@show err
@test_approx_eq err[:re]  0



### test MKdata

using Watershed
using EMIRT
using HDF5
using Base.Test


"""
# in conda root env
import os
import h5py
import numpy as np
from wmem import utils
datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500";
pred_file = "B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184_probs_vol01+vol06";
h5path_in = os.path.join(datadir, pred_file + '.h5', 'data')
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
grad = np.array(np.absolute(np.gradient(ds_in[:], 1)))
h5file_in.close()
h5path_out = os.path.join(datadir, pred_file + '_absgrad.h5', 'main')
h5file_out, ds_out = utils.h5_write(None, grad.shape, grad.dtype,
                                    h5path_out,
                                    element_size_um=np.insert(elsize, 0, 1),
                                    axislabels=np.insert(axlab, 0, 'c'))
ds_out[:] = grad
h5file_out.close()
"""



datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500";
pred_file = "B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184_probs_vol01+vol06";
aff = h5read(joinpath(datadir, pred_file * "_absgrad.h5"), "main");

low = 0.1;
high = 0.8;
thresholds = Tuple{Int64,Float32}[(8000, 0.0)];
dust_size = 600;
seg, rg = watershed(aff; low = low, high = high, thresholds = thresholds, dust_size = dust_size, is_threshold_relative = true);
h5write(joinpath(datadir, pred_file * "_absgrad_seg.h5"), "seg", seg);

"""
datadir="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
datastem="B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184"
ipf=; ids=data; ods=$ids; args='-i zyx -o xyz';
ipf=_probs_vol01+vol06; ids=data; ods=$ids; args='-i zyx -o xyz';
ipf=_probs_vol01+vol06_absgrad; ids=main; ods=${ids}_vol00; args='-i czyx -o xyzc -D 0 1 1 0 0 1 0 0 1 0 0 1';
pf=_probs_vol01+vol06_absgrad_seg; ids=seg; ods=$ids; args='-i zyx -o xyz -e 0.007 0.007 0.1';
python $scriptdir/wmem/stack2stack.py \
    $datadir/blocks_0500/$datastem$ipf.h5/$ids \
    $datadir/blocks_0500/$datastem${ipf}_$ods.nii.gz $args
"""




datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/zws";
aff = h5read(joinpath(datadir, "B-NT-S10-2f_ROI_00ds7_probs_main_vol00_grad.h5"), "main");

# watershed(aff::Array{Float32,4}; low, high, thresholds, dust_size, is_threshold_relative)
# thresholds=[(merge_size, merge_threshold)]

seg = atomicseg(aff);
h5write(joinpath(datadir, "B-NT-S10-2f_ROI_00ds7_probs_main_vol00_grad.h5"), "atomseg", seg);
seg, rg = watershed(aff; is_threshold_relative = true);
# low 0.1, high 0.8, thresholds Tuple{Int64,Float32}[(800, 0.2)]
# INFO: absolute watershed threshold: low: -0.100817, high: 0.019489, thresholds: Tuple{Int64,Float32}[(800, -0.020919)], dust: 600
# steepestascent...
# divideplateaus...
# findbasins!
# found: 21239023 components
# regiongraph...
# Region graph size: 98651500
# use region graph construction code > julia 0.4
# mergeregions...
# Done with merging
# Done with remapping, total: 5931 regions
# Done with updating the region graph, size: 41675
h5write(joinpath(datadir, "B-NT-S10-2f_ROI_00ds7_probs_main_vol00_grad.h5"), "seg", seg);


seg, rg = watershed(aff; low = 0.0, high = 1.0, is_threshold_relative = true);
# INFO: use percentage threshold: low 0.0, high 1.0, thresholds Tuple{Int64,Float32}[(800, -0.020919)]
# INFO: absolute watershed threshold: low: -0.49285802, high: 0.489795, thresholds: Tuple{Int64,Float32}[(800, -0.492858)], dust: 600
# steepestascent...
# divideplateaus...
# findbasins!
# found: 41816208 components
# regiongraph...
# Region graph size: 218970938
# use region graph construction code > julia 0.4
# mergeregions...
# Done with merging
# Done with remapping, total: 27812 regions
# Done with updating the region graph, size: 307919
h5write(joinpath(datadir, "B-NT-S10-2f_ROI_00ds7_probs_main_vol00_grad.h5"), "segall", seg);

seg, rg = watershed(aff; low = 0.0, high = 1.0, dust_size = 10, is_threshold_relative = true);
# INFO: use percentage threshold: low 0.0, high 1.0, thresholds Tuple{Int64,Float32}[(800, -0.492858)]
# INFO: absolute watershed threshold: low: -0.49285802, high: 0.489795, thresholds: Tuple{Int64,Float32}[(800, -0.492858)], dust: 10
# steepestascent...
# divideplateaus...
# findbasins!
# found: 41816208 components
# regiongraph...
# Region graph size: 218970938
# use region graph construction code > julia 0.4
# mergeregions...
# Done with merging
# Done with remapping, total: 27812 regions
# Done with updating the region graph, size: 307919
h5write(joinpath(datadir, "B-NT-S10-2f_ROI_00ds7_probs_main_vol00_grad.h5"), "segdust", seg);

seg, rg = watershed(aff; low = 0.0, high = 1.0, thresholds = Tuple{Int64,Float32}[(800, 0.2)], dust_size = 10, is_threshold_relative = true);
# INFO: use percentage threshold: low 0.0, high 1.0, thresholds Tuple{Int64,Float32}[(800, 0.2)]
# INFO: absolute watershed threshold: low: -0.49285802, high: 0.489795, thresholds: Tuple{Int64,Float32}[(800, -0.020919)], dust: 10
# steepestascent...
# divideplateaus...
# findbasins!
# found: 41816208 components
# regiongraph...
# Region graph size: 218970938
# use region graph construction code > julia 0.4
# mergeregions...
# Done with merging
# Done with remapping, total: 27812 regions
# Done with updating the region graph, size: 307919
h5write(joinpath(datadir, "B-NT-S10-2f_ROI_00ds7_probs_main_vol00_grad.h5"), "segmthr", seg);




low = 0.0;
high = 1.0;
thresholds = Tuple{Int64,Float32}[(100, 0.2)];
dust_size = 10;
seg, rg = watershed(aff; low = low, high = high, thresholds = thresholds, dust_size = dust_size, is_threshold_relative = true);
# INFO: use percentage threshold: low 0.0, high 1.0, thresholds Tuple{Int64,Float32}[(100, 0.2)]
# INFO: absolute watershed threshold: low: -0.49285802, high: 0.489795, thresholds: Tuple{Int64,Float32}[(100, -0.020919)], dust: 10
# steepestascent...
# divideplateaus...
# findbasins!
# found: 41816208 components
# regiongraph...
# Region graph size: 218970938
# use region graph construction code > julia 0.4
# mergeregions...
# Done with merging
# Done with remapping, total: 389363 regions
# Done with updating the region graph, size: 4798320
h5write(joinpath(datadir, "B-NT-S10-2f_ROI_00ds7_probs_main_vol00_grad.h5"), "segmsize", seg);




# new_rg = mergeregions!(seg, rg, counts, thresholds, dust_size)



"""
pf=_probs_main_vol00; ids=main; ods=$ids; args='-i czyx -o xyzc';
pf=_probs_main_vol00_grad; ids=main; ods=${ids}_vol00; args='-i czyx -o xyzc -D 0 1 1 0 0 1 0 0 1 0 0 1';
pf=_probs_main_vol00_grad; ids=atomseg; ods=$ids; args='-i zyx -o xyz -e 0.049 0.049 0.1';
pf=_probs_main_vol00_grad; ids=seg; ods=$ids; args='-i zyx -o xyz -e 0.049 0.049 0.1';
pf=_probs_main_vol00_grad; ids=segall; ods=$ids; args='-i zyx -o xyz -e 0.049 0.049 0.1';
pf=_probs_main_vol00_grad; ids=segdust; ods=$ids; args='-i zyx -o xyz -e 0.049 0.049 0.1';
pf=_probs_main_vol00_grad; ids=segmthr; ods=$ids; args='-i zyx -o xyz -e 0.049 0.049 0.1';
pf=_probs_main_vol00_grad; ids=segmsize; ods=$ids; args='-i zyx -o xyz -e 0.049 0.049 0.1';
python $scriptdir/wmem/stack2stack.py \
    $datadir/zws/$dataset_ds$pf.h5/$ids \
    $datadir/zws/$dataset_ds${pf}_$ods.nii.gz $args
"""
