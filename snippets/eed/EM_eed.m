function EM_eed(datadir, invol, infield, outfield, layers)

% EM_eed('/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU', 'm000_cutout01', '/stack', '/stack', 0)
% EM_eed('/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU', 'm000_cutout01_probs', '/volume/predictions', '/stack', [1,2,3])

addpath(genpath('~/oxscripts/matlab/toolboxes/coherencefilter_version5b'));
addpath('~/oxscripts/matlab/toolboxes/NIFTI_20090703');

for layer = layers;
    
    if layer == 0
        data = h5read([datadir filesep invol '.h5'], infield);
        fname = [datadir filesep invol '_eed2.h5'];
    else
        data = h5read([datadir filesep invol '.h5'], infield, [layer,1,1,1], [1,Inf,Inf,Inf]);
        data = squeeze(data(layer,:,:,:));
        fname = [datadir filesep invol num2str(layer-1) '_eed2.h5'];
    end
    
    u = CoherenceFilter(data, struct('T', 50, 'dt', 1, 'rho', 1, 'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full'));
    h5create(fname, outfield, size(u));
    h5write(fname, outfield, u);
    
end
