host="ndcn0180@arcus-b.arc.ox.ac.uk"
remdir="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNUds7/M3S1GNUds7_processed"
localdir="/Users/michielk"

f="M3S1GNUds7_segmented_large.h5"
f="Ilastik_Mitochondria_within_MF_imported*"
rsync -avz "$host:$remdir/../${f}" $localdir


scriptdir="$HOME/workspace/EM"
datadir="/Users/michielk"

cd ~
xe=0.0511; ye=0.0511; ze=0.05;
datastem='M3S1GNUds7'
pf='_segmented_large'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyxc' -l 'xyzc' &




## copy elsize and labels
elsize = [0.05, 0.0511, 0.0511]
axislabels = 'zyx'
field = 'stack'

infiles = glob.glob(os.path.join(datadir, "{}{}*.h5".format(dset_name, pf)))
for fname in infiles:
    try:
        f = h5py.File(fname, 'a')
        f[field].attrs['element_size_um'] = elsize
        for i, l in enumerate(axislabels):
            f[field].dims[i].label = l
        f.close()
        print("%s done" % fname)
    except:
        print(fname)
