addpath('/Users/michielk/workspace/EM/wmem')
datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S9-2a/blocks';
invol = 'B-NT-S9-2a_00000-01050_00000-01050_00000-00479_probs';
ds_in = '/volume/predictions';
ds_out = '/probs_eed';
EM_eed(datadir, invol, ds_in, ds_out, 1, 1, 1, 1, 184, 200, 200, 2, 0, 20, 20, 5, 1);

% 185x200x200: 0.5GB, 1s / iter  ARC: 3s / iter
% 185x500x500: 3GB, 10s / iter  ARC: 80s / iter
% 185x1000x1000: 10GB, 50s / iter
% 479x500x500: 8GB, 30s / iter

% there are crashes of the compiled routine

