from skimage.segmentation import relabel_sequential
from skimage.morphology import remove_small_objects, binary_opening, ball, closing, binary_erosion, label, relabel_sequential
from skimage.measure import regionprops
from scipy.ndimage.filters import gaussian_filter

datadir='/Users/michielk/M3_S1_GNU_NP/train'
dset_name='m000_01000-01500_01000-01500_00030-00460'

datadir='/Users/michielk/M3_S1_GNU_NP/test'
dset_name='m000_02000-03000_02000-03000_00030-00460'
dset_name='m000_03000-04000_03000-04000_00030-00460'

data, elsize = loadh5(datadir, dset_name)
datamask = data != 0

th = 0.02
prob_myel = loadh5(datadir, dset_name + '_probs0_eed2', fieldname='stack')[0]

# th =
# probs = loadh5(datadir, dset_name + '_probs', fieldname='volume/predictions')[0]
# prob_myel = probs[:,:,:,0] + probs[:,:,:,4]
# prob_myel = gaussian_filter(prob_myel, [0.146, 1, 1])
# writeh5(prob_myel, datadir, dset_name + "_probmyel", element_size_um=elsize, dtype='float')

myelin = np.logical_or(binary_dilation(prob_myel > th), ~datamask)  #mask =  binary_dilation(emyel, ball(5))
remove_small_objects(myelin, min_size=100000, in_place=True)
writeh5(myelin, datadir, dset_name + "_Emyelin" + str(th), element_size_um=elsize)

labels = label(~myelin, return_num=False, connectivity=None)
remove_small_objects(labels, min_size=10000, connectivity=1, in_place=True)
rp = regionprops(labels)
areas = [prop.area for prop in rp]
labs = [prop.label for prop in rp]
llab = labs[np.argmax(areas)]
labels[labels==llab] = 0
labels = relabel_sequential(labels)[0]
writeh5(labels, datadir, dset_name + "_Emyelin" + str(th) + "_labels", element_size_um=elsize)


input_watershed='_ws_l0.95_u1.00_s064'
ws = loadh5(datadir, dset_name + input_watershed)[0]

ulabels = np.trim_zeros(np.unique(labels))
for l in ulabels:
    svoxs = np.trim_zeros(np.unique(ws[labels==l]))
    for sv in svoxs:
        ws[ws==sv] = svoxs[0]

writeh5(ws, datadir, dset_name + input_watershed + "_Emyelin" + str(th) + "_labels", element_size_um=elsize)





# from scipy.ndimage import label
# seed_size = 64
# seeds = label(prob_ics>0.02)
# remove_small_objects(seeds, min_size=seed_size, in_place=True)
# seeds = relabel_sequential(seeds)[0]
# MA = watershed(-prob_ics, seeds, mask=np.logical_and(~myelin, datamask))
# writeh5(MA, datadir, dset_name + outpf, element_size_um=elsize, dtype='int32')
