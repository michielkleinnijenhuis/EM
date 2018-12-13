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
