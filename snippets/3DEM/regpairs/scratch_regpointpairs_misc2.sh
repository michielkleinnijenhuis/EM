import sys
from os import path, makedirs
from argparse import ArgumentParser
import pickle
import math
import glob
from random import sample
import numpy as np

from scipy.optimize import minimize
from skimage import transform as tf

def load_pairs(inputdir, regex, npairs=100):
    """Load a previously generated set of pairs."""
    pairfiles = glob.glob(path.join(inputdir, regex))
    pairs = []
    slcnr = 0
    tilenr = 0
    for pairfile in pairfiles:
        p, src, dst, model, w = pickle.load(open(pairfile, 'rb'))
        population = range(0, src.shape[0])
        try:
            pairnrs = sample(population, npairs)
        except ValueError:
            pairnrs = [i for i in population]
            print("TOO LITTLE DATA for pair in %s!" % pairfile)
            print(pairnrs, src[pairnrs, :], dst[pairnrs, :])
        pairs.append((p, src[pairnrs, :], dst[pairnrs, :], model, w))
        slcnr = max(p[0][0], slcnr)
        tilenr = max(p[0][1], tilenr)
    return pairs, slcnr+1, tilenr+1

inputdir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4'
regex='pair*.pickle'
npairs=10

pairs, n_slcs, n_tiles = load_pairs(inputdir, regex, npairs)

pairfile='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4/pair_c1_d4_s0408-t1_s0409-t3.pickle'
p, src, dst, model, w = pickle.load(open(pairfile, 'rb'))
population = range(0, src.shape[0])
try:
    pairnrs = sample(population, npairs)
except ValueError:
    pairnrs = [i for i in population]
    print("TOO LITTLE DATA for pair in %s!" % pairfile)
    print(pairnrs, src[pairnrs, :], dst[pairnrs, :])
pairs.append((p, src[pairnrs, :], dst[pairnrs, :], model, w))
slcnr = max(p[0][0], slcnr)
tilenr = max(p[0][1], tilenr)



## failed pairs (nans)
[[387, 0], [388, 3], 'tlbr'],
[[389, 0], [390, 3], 'tlbr'],
[[393, 0], [393, 3], 'tlbr'],
[[395, 0], [396, 3], 'tlbr'],
[[396, 0], [396, 3], 'tlbr'],
[[397, 0], [398, 3], 'tlbr'],
[[401, 0], [402, 3], 'tlbr'],
[[402, 0], [402, 3], 'tlbr'],
[[405, 0], [406, 3], 'tlbr'],
[[406, 0], [406, 3], 'tlbr'],
[[407, 0], [407, 3], 'tlbr'],
[[409, 0], [409, 3], 'tlbr'],
[[411, 0], [411, 3], 'tlbr'],
[[412, 0], [412, 3], 'tlbr'],
[[413, 0], [414, 3], 'tlbr'],
[[414, 0], [415, 3], 'tlbr'],
[[416, 0], [417, 3], 'tlbr'],
[[417, 0], [417, 3], 'tlbr'],
[[420, 0], [421, 3], 'tlbr'],
[[421, 0], [421, 3], 'tlbr'],
[[422, 0], [422, 3], 'tlbr'],
[[423, 0], [424, 3], 'tlbr'],
[[424, 0], [424, 3], 'tlbr'],
[[425, 0], [425, 3], 'tlbr'],
[[426, 0], [427, 3], 'tlbr'],
[[436, 0], [437, 3], 'tlbr'],
[[439, 0], [440, 3], 'tlbr'],
[[441, 0], [442, 3], 'tlbr'],
[[444, 0], [445, 3], 'tlbr'],
[[446, 0], [446, 3], 'tlbr'],
[[446, 0], [447, 3], 'tlbr'],
[[449, 0], [450, 3], 'tlbr'],
[[451, 0], [452, 3], 'tlbr'],
[[458, 0], [459, 3], 'tlbr']]


inputdir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4'
