function EM_eed_simple(datadir, invol, infield, outfield, layer, T, dt, rho)

% EM_eed('/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU', 'm000_cutout01', '/stack', '/stack', 0)
% EM_eed('/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU', 'm000_cutout01_probs', '/volume/predictions', '/stack', 1)

% make sure layer arg are numeric
if isdeployed
    layer = str2num(layer);
    T = str2double(T);
    dt = str2double(dt);
    rho = str2double(rho);
else
    addpath(genpath('~/oxscripts/matlab/toolboxes/coherencefilter_version5b'));
end

stackinfo = h5info([datadir filesep invol '.h5'], infield);

if layer == 0
    data = h5read([datadir filesep invol '.h5'], infield);
    fname = [datadir filesep invol '_eed2.h5'];
    cs = stackinfo.ChunkSize;
    es = stackinfo.Attributes(1).Value;
    al = char(stackinfo.Attributes(2).Value)';
else
    data = h5read([datadir filesep invol '.h5'], infield, [layer,1,1,1], [1,Inf,Inf,Inf]);
    data = squeeze(data(1,:,:,:));
    fname = [datadir filesep invol num2str(layer-1) '_eed2.h5'];
    cs = stackinfo.ChunkSize(2:4);
    es = stackinfo.Attributes(1).Value(2:4);
    al = char(stackinfo.Attributes(2).Value(2:4))';
end

u = CoherenceFilter(data, struct('T', T, 'dt', dt, 'rho', rho, 'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full'));
h5create(fname, outfield, size(u), 'Deflate', 4, 'Chunksize', cs);
h5write(fname, outfield, u);

h5writeatt(fname, outfield, stackinfo.Attributes(1).Name, es)
h5writeatt(fname, outfield, stackinfo.Attributes(2).Name, al)
