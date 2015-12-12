addpath(genpath('/Users/michielk/oxscripts/matlab/toolboxes/coherencefilter_version5b'));
addpath /Users/michielk/oxscripts/matlab/toolboxes/NIFTI_20090703

datadir='/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/pipeline_test';
invol = 'pixprob_training_probs';
invol = 'segclass_training_probs';
data = h5read([datadir filesep invol '.h5'], '/volume/predictions');

%%
layer = 3;
image = squeeze(data(layer,:,:,:));

u = CoherenceFilter(image, struct('T',50,'dt',1,'rho',1,'Scheme','R','eigenmode',2,'verbose','full'));
h5create([datadir filesep invol num2str(layer-1) '_eed2.h5'], '/stack', size(u));
h5write([datadir filesep invol num2str(layer-1) '_eed2.h5'], '/stack', u);

%%
layer = [3 6];
image = squeeze(mean(data(layer,:,:,:),1));
image = squeeze(max(data(layer,:,:,:),[],1));
u = CoherenceFilter(image, struct('T',50,'dt',1,'rho',1,'Scheme','R','eigenmode',2,'verbose','full'));
h5create([datadir filesep invol '36_eed2.h5'], '/stack', size(u));
h5write([datadir filesep invol '36_eed2.h5'], '/stack', u);


%%
datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU';
invol = 'm000_cutout01_probs';
data = h5read([datadir filesep invol '.h5'], '/volume/predictions');

for layer = 1:3;
    image = squeeze(data(layer,:,:,:));
    u = CoherenceFilter(image, struct('T',50,'dt',1,'rho',1,'Scheme','R','eigenmode',2,'verbose','full'));
    h5create([datadir filesep invol num2str(layer-1) '_eed2.h5'], '/stack', size(u));
    h5write([datadir filesep invol num2str(layer-1) '_eed2.h5'], '/stack', u);
end

%%

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU';
invol = 'm000_cutout01';
data = h5read([datadir filesep invol '.h5'], '/stack');

image = squeeze(data(:,:,:));
u = CoherenceFilter(image, struct('T',50,'dt',1,'rho',1,'Scheme','R','eigenmode',2,'verbose','full'));
h5create([datadir filesep invol '_eed2.h5'], '/stack', size(u));
h5write([datadir filesep invol '_eed2.h5'], '/stack', u);
