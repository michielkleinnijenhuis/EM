datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/old';
dataset = 'B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184';
dpf = '_probs_vol00+vol02+vol04+vol07';
invol = [dataset, dpf];
infield = '/data';
outfield = '/probs_eed';

addpath(genpath('~/workspace/EM/wmem'));

EM_eed_simple(datadir, invol, infield, outfield, 0, 1, 1, 1);


%%
addpath('/Users/michielk/workspace/EM/wmem')
datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500';
dataset = 'B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184';

dpf = '_probs';
invol = [dataset, dpf];
outvol = [dataset, dpf, '_eed'];

ds_in = '/volume/predictions';
ds_out = '/probs_eed';
EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, 0, 1, 1, 1);

ds_in = '/sum0247';
ds_out = '/sum0247_eed';
EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, 0, 1, 1, 1);

ds_in = '/sum16';
ds_out = '/sum16_eed';
EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, 0, 1, 1, 1);

% ds_out = ['/probs_eed'];
% EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, [1:3], 1, 1, 1);

% ds_out = ['/probs6_eed'];
% EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, 7, 1, 1, 1);


% dpf = '_probs_vol00+vol02+vol04+vol07';
% invol = [dataset, dpf];
% outvol = [dataset, dpf, '_eed'];
% ds_in = '/data';
% 
% ds_out = ['/probs_eed'];
% EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, 0, 1, 1, 1);


%%

stackinfo = h5info([datadir filesep invol '.h5'], ds_in);

elsize_index = find(strcmp({stackinfo.Attributes.Name}, 'element_size_um')==1);
el = stackinfo.Attributes(elsize_index).Value;
axlab_index = find(strcmp({stackinfo.Attributes.Name}, 'DIMENSION_LABELS')==1);
al = stackinfo.Attributes(axlab_index).Value;
cdim_idx = find(strcmp( cellfun( @(sas) sas, al, 'uni', false ), {'c'} ))
cs = stackinfo.ChunkSize;
cs(5 - cdim_idx) = [];

starts = [1, 1, 1, 1];
starts(5 - cdim_idx) = layer;
counts = [Inf, Inf, Inf, Inf];
counts(5 - cdim_idx) = 1;

data = h5read([datadir filesep invol '.h5'], ds_in, starts, counts);
data = squeeze(data);

% u = [];
% u3d = CoherenceFilter(data, struct('T', 1, 'dt', 1, 'rho', 1, ...
%     'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full'));
% 
% u = cat(4, u, u3d);

% stackinfo.Attributes.Name

%%
cdim_idx_data = 1
perm = [1, 2, 3];
insert = @(a, x, n)cat(2,  x(1:n-1), a, x(n:end));
insert(4, perm, cdim_idx_data)

%%

A = magic(3); B = pascal(3);
C = cat(4, A, B);


