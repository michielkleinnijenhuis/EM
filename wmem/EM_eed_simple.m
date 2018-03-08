function u = EM_eed_simple(datadir, ...
    infile, infield, ...
    outfile, outfield, ...
    vol_idxs, ...
    T, dt, rho)

% make sure args are numeric
if isdeployed
    vol_idxs = eval(vol_idxs);
    T = str2double(T);
    dt = str2double(dt);
    rho = str2double(rho);
else
    addpath(genpath('~/oxscripts/matlab/toolboxes/coherencefilter_version5b'));
end

inputfile = [datadir filesep infile '.h5'];
outputfile = [datadir filesep outfile '.h5'];

% find attributes
stackinfo = h5info(inputfile, infield);
outsize = stackinfo.Dataspace.Size;
ndim = length(stackinfo.Dataspace.Size);
es_idx = find(strcmp({stackinfo.Attributes.Name}, 'element_size_um')==1);
es = stackinfo.Attributes(es_idx).Value;
al_idx = find(strcmp({stackinfo.Attributes.Name}, 'DIMENSION_LABELS')==1);
al = stackinfo.Attributes(al_idx).Value;
cs = stackinfo.ChunkSize;


if ndim == 4

    % find the volume dimension index
    cdim_idx = find(strcmp( cellfun( @(sas) sas, al, 'uni', false ), {'c'} ));
    cdim_idx_data = 5 - cdim_idx;

    % set vol_idxs to the full range
    if vol_idxs == 0
        vol_idxs = 1:stackinfo.Dataspace.Size(cdim_idx_data);
    end

    % make 3D by removing cdim
    if length(vol_idxs) == 1
        outsize(cdim_idx_data) = [];
        cs(cdim_idx_data) = [];
        es(cdim_idx) = [];
        al(cdim_idx) = [];
    end

end

% create the dataset
h5create(outputfile, outfield, outsize, 'Deflate', 4, 'Chunksize', cs);
h5writeatt(outputfile, outfield, stackinfo.Attributes(es_idx).Name, es)
EM_eed_writeattr(inputfile, infield, outputfile, outfield, al)

% read and filter the data % TODO: 2D/5D?
if ndim == 3

    data = h5read(inputfile, infield);

    u = CoherenceFilter(data, struct('T', T, 'dt', dt, 'rho', rho, ...
        'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full'));

elseif ndim == 4

    u = [];
    for layer = vol_idxs

        % always read only a 3D volume to keep mem down
        starts = [1, 1, 1, 1];
        starts(cdim_idx_data) = layer;
        counts = [Inf, Inf, Inf, Inf];
        counts(cdim_idx_data) = 1;
        data = h5read(inputfile, infield, starts, counts);
        data = squeeze(data);

        % filter 3D volume
        u3d = CoherenceFilter(data, struct('T', T, 'dt', dt, 'rho', rho, ...
            'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full')); % xyz
        u = cat(4, u, u3d); % xyzc

    end
    
    % c needs to go from dim=4 to dim=cdim_idx_data
    if length(vol_idxs) > 1
        perm = [1, 2, 3];
        insert = @(a, x, n)cat(2,  x(1:n-1), a, x(n:end));
        perm = insert(4, perm, cdim_idx_data);
        u = permute(u, perm);
    end
    
end

% write the filtered volume(s) and attributes
h5write(outputfile, outfield, u);


function EM_eed_writeattr(inputfile, infield, outputfile, outfield, axlab)


fid = H5F.open(inputfile);
gid = H5G.open(fid, '/');
did = H5D.open(gid, infield);
aid = H5A.open(did, 'DIMENSION_LABELS');

tid = H5A.get_type(aid);
sid = H5A.get_space(aid);
H5S.set_extent_simple(sid, 1, length(axlab), 5)

H5A.close(aid);
H5D.close(did);
H5G.close(gid);
H5F.close(fid);


fid = H5F.open(outputfile, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
gid = H5G.open(fid, '/');
did = H5D.open(gid, outfield);

pid = H5P.create('H5P_ATTRIBUTE_CREATE');
aid = H5A.create(did, 'DIMENSION_LABELS', tid, sid, pid);
H5A.write(aid, tid, axlab)

H5A.close(aid);
H5D.close(did);
H5G.close(gid);
H5F.close(fid);
