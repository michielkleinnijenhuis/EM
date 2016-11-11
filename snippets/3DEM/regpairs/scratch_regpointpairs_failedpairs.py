import os
import pickle

outputdir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU_regpointpairs/local/reg_d4"
outputdir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_regpointpairs/reg_d4"
outputdir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4"
downsample_factor = 4
# outputdir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_regpointpairs/reg"
# downsample_factor = 1
n_slcs = 100
offsets = 1

# load unique pairs
pairstring = 'unique_pairs' + '_c' + str(offsets) + '_d' + str(downsample_factor)
pairfile = os.path.join(outputdir, pairstring + '.pickle')
with open(pairfile, 'rb') as f:
    unique_pairs = pickle.load(f)

# check which pairs failed
failed_pairs = []
for p in unique_pairs:
    pairstring = 'pair' + \
                 '_c' + str(offsets) + \
                 '_d' + str(downsample_factor) + \
                 '_s' + str(p[0][0]).zfill(4) + \
                 '-t' + str(p[0][1]) + \
                 '_s' + str(p[1][0]).zfill(4) + \
                 '-t' + str(p[1][1])
    try:
        f = open(os.path.join(outputdir, pairstring + ".pickle"), 'rb')
    except:
        failed_pairs.append(p)

# save lost_pairs
pairstring = 'failed_pairs' + '_c' + str(offsets) + '_d' + str(downsample_factor)
pairfile = os.path.join(outputdir, pairstring + '.pickle')
with open(pairfile, 'wb') as f:
    pickle.dump(failed_pairs, f)
