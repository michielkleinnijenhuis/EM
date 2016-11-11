from skimage.morphology import remove_small_objects, binary_opening, ball, closing, binary_erosion, label, relabel_sequential
from skimage.measure import regionprops

scriptdir="$HOME/workspace/EM"
datadir='/Users/michielk/M3_S1_GNU_NP/train'
dset_name='m000_01000-01500_01000-01500_00030-00460'
datastem='m000_01000-01500_01000-01500_00030-00460'
data, elsize = loadh5(datadir, dset_name)
datamask = data != 0
# datamask = binary_dilation(binary_fill_holes(datamask))  # TODO

# probs = loadh5(datadir, dset_name + '_probs', fieldname='volume/predictions')[0]
# myelin = np.logical_and(probs[:,:,:,0] > 0.5, datamask)
probs = loadh5(datadir, dset_name + '_probs0_eed2', fieldname='stack')[0]
myelin = np.logical_and(probs > 0.1, datamask)
remove_small_objects(myelin, min_size=100000, in_place=True)
writeh5(myelin, datadir, dset_name + "_Emyelin01", element_size_um=elsize)
emyel = binary_erosion(myelin)
remove_small_objects(emyel, min_size=100000, in_place=True)
writeh5(emyel, datadir, dset_name + "_Emyelin05_ero", element_size_um=elsize)
dmyel = binary_dilation(emyel, ball(5))
writeh5(dmyel, datadir, dset_name + "_Emyelin05_erodil", element_size_um=elsize)


prob_myel = loadh5(datadir, dset_name + '_probs0_eed2')[0]
pmyel = closing(prob_myel, selem=ball(3))
writeh5(pmyel, datadir, dset_name + "_probs0_eed2_closing3", element_size_um=elsize)
myelin = np.logical_and(prob_myel > 0.2, datamask)
emyel = binary_erosion(myelin)
writeh5(emyel, datadir, dset_name + "_myelin_ero", element_size_um=elsize)
labels = label(emyel)
rp = regionprops(labels)
areas = [prop.area for prop in rp]

i = np.argmax(areas)
labels[labels==i] = 0
dmyel = binary_dilation(labels)
writeh5(dmyel, datadir, dset_name + "_dmyelin", element_size_um=elsize)
# binary_opening(myelin, selem=ball(3), out=myelin)
remove_small_objects(myelin, min_size=100000, in_place=True)
writeh5(myelin, datadir, dset_name + "_myelin", element_size_um=elsize)

prob_mito = loadh5(datadir, dset_name + '_probs2_eed2')[0]
mito = prob_mito > 0.3
writeh5(mito, datadir, dset_name + "_mito", element_size_um=elsize)

prob_mm = prob_myel + prob_mito
writeh5(prob_mm, datadir, dset_name + "_probs02_eed2", element_size_um=elsize, dtype='float')


pf=""
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' -u
pf='_myelin'
pf='_mito'
pf='_probs0_eed2'
pf='_probs0_eed2_closing3'
pf='_myelin_ero3'
pf='_probs02_eed2'
pf='_dmyel'
pf='_probs3_eed2'
pf='_probs2_eed2'
pf='_probs5_eed2'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'
pf='_probs'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 1 -i 'zyxc' -l 'xyzc' -f '/volume/predictions'
