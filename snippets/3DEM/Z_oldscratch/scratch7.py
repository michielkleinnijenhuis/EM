scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_01000-02000_01000-02000_00030-00460.h5" \
"${datadir}/${dataset}_01000-02000_01000-02000_00030-00460.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
dataset='m000_01000-02000_01000-02000_00030-00460'
dataset='m000_01000-02000_02000-03000_00030-00460'
pf='_seg'
pf='_seeds_MA'
pf='_probs_ws_MA'
pf='_probs_ws_MAfilled'
pf='_probs_ws_MMdistsum_distfilter'
pf='_probs_ws_UA'
pf='_probs_ws_PA'
pf='_seeds_UA'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}${pf}.h5" \
"${datadir}/${dataset}${pf}.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_probs_ws_MA_probs_ws_MA_manseg.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs_ws_M* /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_seg.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_seeds_MA.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs_ws_M*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs_ws_?A.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_05000-06000_00000-01000_?????-?????_probs.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
