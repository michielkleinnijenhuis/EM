clear all; clc;

conf.toolboxdir = '~/oxscripts/matlab/toolboxes';
conf.scriptdir = '~/workspace/DifSim/utilities';

% addpath(genpath([conf.toolboxdir filesep 'geom2d-2015.05.13']));
addpath([conf.scriptdir filesep 'geoms/3DEM']);
addpath([conf.toolboxdir filesep 'iso2mesh']);
addpath([conf.toolboxdir filesep 'strel3d']);
addpath([conf.toolboxdir filesep 'imMinkowski_2015.04.20/imMinkowski']);

datadir = '/Users/michielk/oxdata/P01/EM/Kashturi11';

xrange = [5000,6000]; yrange = [7500,8500]; zrange = [1000,1100];
% xrange = [5500,6000]; yrange = [7500,8000]; zrange = [1000,1100];
% xrange = [5000,5500]; yrange = [7500,8000]; zrange = [1000,1100];
% xrange = [5500,6000]; yrange = [8000,8500]; zrange = [1000,1100];
xrange = [5000,5500]; yrange = [8000,8500]; zrange = [1000,1100];
xrange = [5000,5500]; yrange = [8000,8500]; zrange = [1100,1200]; conf.cu = 'cutout01'; conf.resno = 1; conf.anno = 'kat11segments';
% xrange = [5000,5500]; yrange = [8000,8500]; zrange = [1000,1380]; conf.cu = 'cutout02'; conf.resno = 1; conf.anno = 'kat11segments';
% xrange = [2750,7500]; yrange = [6750,10000]; zrange = [1000,1380];  % 4750x3250x380

% xrange = [11000,11500]; yrange = [13000,13500]; zrange = [1000,1255]; conf.cu = 'ac3'; conf.resno = 0;  conf.anno = 'ac3';% [10943 17426; 12993 19471; 1000 1255]
% xrange = [5500,6000]; yrange = [6500,7000]; zrange = [1000,1255]; conf.cu = 'ac3'; conf.resno = 1;  conf.anno = 'ac3';

if conf.resno == 1
    conf.voxelsize = [0.006, 0.006, 0.03];
end
conf.conn = 26;

mkdir(datadir, conf.cu);
mkdir([datadir filesep conf.cu], 'binsurf');
mkdir([datadir filesep conf.cu], 'cgalsurf');
mkdir([datadir filesep conf.cu], 'isosurf');

%% get the data and axon annotation

oo = OCP();
oo.setServerLocation('http://openconnecto.me');
oo.setImageToken('kasthuri11cc');
oo.setAnnoToken(conf.anno);

q = OCPQuery;
q.setType(eOCPQueryType.annoDense);
q.setCutoutArgs(xrange,yrange,zrange,conf.resno);
anno = oo.query(q);

q = OCPQuery;
q.setType(eOCPQueryType.imageDense);
q.setCutoutArgs(xrange,yrange,zrange,conf.resno);
im = oo.query(q);

% Visualize results
h = image(im); h.associate(anno);

%%

ds = 1;
conf.voxelsize = conf.voxelsize .* [ds ds 1];
conf.xyzOffset = anno.xyzOffset ./ [ds ds 1];
labelimage = anno.data(1:ds:end,1:ds:end,:);  

%%

labels = unique(labelimage(labelimage>0));
% labels = labels(10);
conf.nvoxelthreshold = int16(500 / ds^2);

% conf.method = 'binsurf'; conf.opt = [];
conf.method = 'isosurf'; conf.opt = [];
% conf.method = 'cgalsurf'; conf.opt.maxnode = 100000; conf.opt.radbound = 5;
% conf.method = 'simplify'; conf.opt.keepratio = 1;

[smalllabels] = anno2mesh([datadir filesep conf.cu], labelimage, ...
    labels, conf.method, conf.opt, conf.voxelsize, conf.xyzOffset, ...
    conf.nvoxelthreshold, conf.conn)

save([datadir filesep conf.cu filesep conf.method filesep 'data.mat']);

%% create 3D ECS container

conf.bbox.margin = 0.1;
conf.bbox.shapes.upsamplefactor = 1;

idxs.nonvalid = smalllabels;
Lb = labelimage;
mask = Lb>0;
for i = idxs.nonvalid
    mask(Lb==i) = false;
end

[no, el] = shaped_boundary_3DEM(mask, ...
    conf.bbox.margin, conf.bbox.shapes.upsamplefactor, conf.voxelsize, ...
    conf.method, conf.opt);

clear nomu
for ii = 1:3
    nomu(:,ii) = no(:,ii) * conf.voxelsize(ii) + ...
        anno.xyzOffset(ii) * conf.voxelsize(ii);
end
savebinstl(nomu, el, [datadir filesep conf.cu filesep 'stl_cgal' filesep ...
    'ECS.' num2str(0, '%05d') '.00.stl'], num2str(i));

% orphans = unique(Lb(~mask))';
% orphans(orphans==0) = [];

%%













%% SCRATCH ...

%% isosurf

label = labels(10)

strelsize = 3;
se = ones([strelsize,strelsize,strelsize]);
D = padarray(labelimage, [strelsize,strelsize,strelsize]);

labelmask = D == label;

[no,el] = isosurface(labelmask,0.5);

figure, patch(isosurface(labelmask,0.5))

%% save as nifti
addpath /Users/michielk/oxscripts/matlab/toolboxes/NIFTI_20090703

mvol1 = nii_zip('~/workspace/DifSim/utilities/geoms/2DEM/M3_S1_GNU/0250_m000/0250_m000_seg.nii');

ds = 4;
mvol1.img = int16(im.data(1:ds:end,1:ds:end,:));
mvol1.hdr.dime.dim(2:4) = size(mvol1.img);
mvol1.hdr.dime.pixdim(2:4) = conf.voxelsize .* [ds ds 1];
save_untouch_nii(mvol1, [datadir filesep conf.cu filesep 'niftis/data.nii'])
mvol1.img = int16(anno.data(1:ds:end,1:ds:end,:));
mvol1.hdr.dime.dim(2:4) = size(mvol1.img);
mvol1.hdr.dime.pixdim(2:4) = conf.voxelsize .* [ds ds 1];
save_untouch_nii(mvol1, [datadir filesep conf.cu filesep 'niftis/anno.nii'])
labelmask = anno.data(1:ds:end,1:ds:end,:) == 862;
mvol1.img = int16(labelmask);
mvol1.hdr.dime.dim(2:4) = size(mvol1.img);
mvol1.hdr.dime.pixdim(2:4) = conf.voxelsize .* [ds ds 1];
save_untouch_nii(mvol1, [datadir filesep conf.cu filesep 'niftis/label862.nii'])

mvol1.img = int16(im.data);
mvol1.hdr.dime.dim(2:4) = size(mvol1.img);
mvol1.hdr.dime.pixdim(2:4) = conf.voxelsize;
save_untouch_nii(mvol1, [datadir filesep conf.cu filesep 'niftis/fulldata.nii'])
mvol1.img = int16(anno.data);
mvol1.hdr.dime.dim(2:4) = size(mvol1.img);
mvol1.hdr.dime.pixdim(2:4) = conf.voxelsize;
save_untouch_nii(mvol1, [datadir filesep conf.cu filesep 'niftis/fullanno.nii'])
labelmask = anno.data == 862;
mvol1.img = int16(labelmask);
mvol1.hdr.dime.dim(2:4) = size(mvol1.img);
mvol1.hdr.dime.pixdim(2:4) = conf.voxelsize;
save_untouch_nii(mvol1, [datadir filesep conf.cu filesep 'niftis/fulllabel862.nii'])

%%

im = [0 0 0 1 1 1; 0 0 0 1 1 1; 0 0 0 1 1 1; ...
      1 1 1 0 0 0; 1 1 1 0 0 0; 1 1 1 0 0 0;]

se = [1 1 1; 1 1 1; 1 1 1];
im2 = imerode(~im, se)
se = [0 1 0; 1 1 1; 0 1 0];
im3 = imdilate(im2, se)
~im3


se = [0 1 0; 1 1 1; 0 1 0];
im2 = imdilate(im, se)
se = [1 1 1; 1 1 1; 1 1 1];
im3 = imerode(im2, se)

%%


strelsize = 3;
se = ones([strelsize,strelsize,strelsize]);
D = padarray(anno.data, [strelsize,strelsize,strelsize]);
labelmask = D == 15;

% [offsets, heights] = getneighbors(double(labelmask));


%% enforce ECS % might want to do this on the 'full' image
labelimage = anno.data(1:ds:end,1:ds:end,:);

% first do a watershed to fill the volume


showslice = 20;
padsize = 3;
L = padarray(labelimage, [padsize,padsize,padsize]);
M0 = L==0; 
M = L==0;
figure; plotbrowser;
imshow(M0(:,:,showslice)); hold on;

strelsize = 3;
se = strel(ones([strelsize,strelsize,strelsize]));
[connc, heights] = getneighbors(se);
connc = connc(1:12,:);
% connc = [-1 -1 -1; -1 -1  0; -1 -1  1; -1  0 -1; -1  0  0; -1  0  1; -1  1 -1; -1  1  0; -1  1  1; ...
%           0 -1 -1;  0 -1  0;  0 -1  1;  0  0 -1;            0  0  1;  0  1 -1;  0  1  0;  0  1  1; ...
%           1 -1 -1;  1 -1  0;  1 -1  1;  1  0 -1;  1  0  0;  1  0  1;  1  1 -1;  1  1  0;  1  1  1]; % 26-conn
% connc = [          -1 -1  0;           -1  0 -1; -1  0  0; -1  0  1;           -1  1  0          ; ...
%           0 -1 -1;  0 -1  0;  0 -1  1;  0  0 -1;            0  0  1;  0  1 -1;  0  1  0;  0  1  1; ...
%                     1 -1  0;            1  0 -1;  1  0  0;  1  0  1;            1  1  0          ]; % 18-conn
% connc = [                                        -1  0  0;                                         ...
%                     0 -1  0;            0  0 -1;            0  0  1;            0  1  0;           ...
%                                                   1  0  0;                                       ]; % 6-conn
% se(:,:,1) = [0 1 0;1,1,1;0,1,0]; se(:,:,2) = [1 1 1;1,1,1;1,1,1]; se(:,:,3) = [0 1 0;1,1,1;0,1,0]; % 18-conn
% se(:,:,1) = [0 0 0;0,1,0;0,0,0]; se(:,:,2) = [0 1 0;1,1,1;0,1,0]; se(:,:,3) = [0 0 0;0,1,0;0,0,0]; % 6-conn
for i = 1:size(connc, 1)
    K = circshift(L, [connc(i,1), connc(i,2), connc(i,3)]);
    M = M | (L ~= K & (L ~= 0 & K ~= 0));
end
L(M) = 0;
imshow(M(:,:,showslice)); % LM = L==0; LM(~M) = 1; figure; imshow(LM(:,:,showslice));
labelimage = L(padsize+1:end-padsize,padsize+1:end-padsize,padsize+1:end-padsize);

mvol1.img = int16(labelimage);
mvol1.hdr.dime.dim(2:4) = size(mvol1.img);
mvol1.hdr.dime.pixdim(2:4) = conf.voxelsize;
save_untouch_nii(mvol1, [datadir filesep conf.cu filesep 'niftis/labelimage_enforceECS26.nii'])

% labels = unique(L(L>0));
% [smalllabels] = anno2mesh([datadir filesep conf.cu], L, ...
%     labels, 'binsurfaces', [], conf.voxelsize, anno.xyzOffset, 500, 26)
% save([datadir filesep conf.cu filesep 'data.mat'])

%%
%% function to detect edge-connections

% only apply to bordervoxels?

% connc = [-1 -1 -1; -1 -1  0; -1 -1  1; -1  0 -1;           -1  0  1; -1  1 -1; -1  1  0; -1  1  1; ...
%           0 -1 -1;            0 -1  1;                                0  1 -1;            0  1  1; ...
%           1 -1 -1;  1 -1  0;  1 -1  1;  1  0 -1;            1  0  1;  1  1 -1;  1  1  0;  1  1  1]; % ~6-conn

% cv = L(101,101,51);
% nh = L(100:102,100:102,50:52);
% obidx = nh == cv;


% define intervals
% define intervals for removing edge-connected voxels (12 candidate configurations)
inter(:,:,1,1) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,2,1) = [ 1 -1  0; -1  1  0;  0  0  0;];
inter(:,:,3,1) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,1,2) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,2,2) = [ 0 -1  1;  0  1 -1;  0  0  0;];
inter(:,:,3,2) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,1,3) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,2,3) = [ 0  0  0;  0  1 -1;  0 -1  1;];
inter(:,:,3,3) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,1,4) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,2,4) = [ 0  0  0; -1  1  0;  1 -1  0;];
inter(:,:,3,4) = [ 0  0  0;  0  0  0;  0  0  0;]; % z-plane
inter(:,:,1,5) = [ 0  0  0;  0 -1  1;  0  0  0;];
inter(:,:,2,5) = [ 0  0  0;  0  1 -1;  0  0  0;];
inter(:,:,3,5) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,1,6) = [ 0  0  0;  1 -1  0;  0  0  0;];
inter(:,:,2,6) = [ 0  0  0; -1  1  0;  0  0  0;];
inter(:,:,3,6) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,1,7) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,2,7) = [ 0  0  0; -1  1  0;  0  0  0;];
inter(:,:,3,7) = [ 0  0  0;  1 -1  0;  0  0  0;];
inter(:,:,1,8) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,2,8) = [ 0  0  0;  0  1 -1;  0  0  0;];
inter(:,:,3,8) = [ 0  0  0;  0 -1  1;  0  0  0;]; % y-plane
inter(:,:,1,9) = [ 0  1  0;  0 -1  0;  0  0  0;];
inter(:,:,2,9) = [ 0 -1  0;  0  1  0;  0  0  0;];
inter(:,:,3,9) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,1,10) = [ 0  0  0;  0 -1  0;  0  1  0;];
inter(:,:,2,10) = [ 0  0  0;  0  1  0;  0 -1  0;];
inter(:,:,3,10) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,1,11) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,2,11) = [ 0  0  0;  0  1  0;  0 -1  0;];
inter(:,:,3,11) = [ 0  0  0;  0 -1  0;  0  1  0;];
inter(:,:,1,12) = [ 0  0  0;  0  0  0;  0  0  0;];
inter(:,:,2,12) = [ 0 -1  0;  0  1  0;  0  0  0;];
inter(:,:,3,12) = [ 0  1  0;  0 -1  0;  0  0  0;]; % x-plane

% select object
D = padarray(anno.data, [3,3,3]);
labelmask = D == 862;
% correct object
for i = 1:size(inter,4)
    labelmask(bwhitmiss(labelmask,inter(:,:,:,i))) = 0;
end


% write mesh

labelmask = imfill(labelmask, se, 'holes');
[no,el] = binsurface(labelmask, 3);
[no, el] = meshcheckrepair(no, el, 'deep');
for ii = 1:3
    no(:,ii) = no(:,ii) - 3 * conf.voxelsize(ii); % correct for padding
    no(:,ii) = no(:,ii) * conf.voxelsize(ii) + ...
        anno.xyzOffset(ii) * conf.voxelsize(ii); % convert to mu and translate to offset
end
savebinstl(no, el, [datadir filesep 'test.stl'], num2str(15));


%%

slc = 85;
figure; plotbrowser;
imshow(~labelmask(:,:,slc)); hold on;
imshow(imdilate(~labelmask(:,:,slc), se)); hold on;

CHI = imEuler3d(labelmask, 26);

lb = imbothat(labelmask, se); imshow(lb(:,:,slc)); hold on;
lb = imtophat(labelmask, se); imshow(lb(:,:,slc)); hold on;
lb = imtophat(labelmask, se); imshow(lb(:,:,slc)); hold on;


%% for binsurfaces (will result in non-manifold objects (if not pre-processed))

dims = size(anno.data);
conf.voxelsize = [0.006, 0.006, 0.03];
labels = unique(anno.data(anno.data>0));

%%% with binsurface
se = ones([3,3,3]);
D = padarray(anno.data, [3 3 3]);
failinglabels = [];
for i = labels'
    clear labelmask nomu
    labelmask = D == i;
    if nnz(labelmask) > 500
%         labelmask = imopen(labelmask, se);
        labelmask = imdilate(labelmask, se);
        [no,el] = binsurface(labelmask, 3);
        [no, el] = meshcheckrepair(no, el, 'deep');
        for ii = 1:3
            no(:,ii) = no(:,ii) - 3*conf.voxelsize(ii); % correct for padding
            nomu(:,ii) = no(:,ii) * conf.voxelsize(ii) + ...
                anno.xyzOffset(ii) * conf.voxelsize(ii);
        end
        savebinstl(nomu, el, [datadir filesep conf.cu filesep 'stl_binsurfaces' filesep ...
            'U_A.' num2str(i, '%05d') '.01.stl'], num2str(i));
    else
        failinglabels = [failinglabels i];
    end
end

save([datadir filesep conf.cu filesep 'data_binsurfaces.mat'])

%%

dims = size(anno.data);
conf.voxelsize = [0.006, 0.006, 0.03];

labels = unique(anno.data(anno.data>0));

%%% with cgal
% failinglabels = [834 3498 3723 4014 4299 4963 5140 5365]; % cutout01: < 500 voxels
% failinglabels = [3133 3270 4116 4148 4153 4297 4549 5127 5812 5914 6097];  % cutout02: < 500 voxels 
% cutout2 hangs on 4297(ii=287) 4549(ii=342) 5127(ii=400) 5812(ii=501) 5914(ii=512) 6097(ii=521)
failinglabels = [];
for i = labels(522:end)'
    clear labelmask nomu
    labelmask = anno.data == i;
    if nnz(labelmask) > 500
        [no,el] = binsurface(labelmask, 3);
        [no, el, ~, ~]=vol2surf(labelmask, 1:dims(1), 1:dims(2), 1:dims(3), ...
            conf.opt, 1, conf.method, 1);
        [no, el] = meshcheckrepair(no, el, 'deep');
        for ii = 1:3
            nomu(:,ii) = no(:,ii) * conf.voxelsize(ii) + ...
                anno.xyzOffset(ii) * conf.voxelsize(ii);
        end
        savebinstl(nomu, el, [datadir filesep conf.cu filesep 'stl_cgal' filesep ...
            'U_A.' num2str(i, '%05d') '.01.stl'], num2str(i));
    else
        failinglabels = [failinglabels i];
    end
end

save([datadir filesep conf.cu filesep 'data.mat'])


%% get cylinders

oo = OCP();
oo.setServerLocation('http://openconnecto.me');
oo.setImageToken('kasthuri11cc');
% oo.setAnnoToken('kat11mojocylinder');
% oo.setAnnoToken('kat11redcylinder');
oo.setAnnoToken('kat11greencylinder');

q = OCPQuery;
q.setType(eOCPQueryType.annoDense);
q.setCutoutArgs(xrange,yrange,zrange,1);
anno = oo.query(q);

q = OCPQuery;
q.setType(eOCPQueryType.imageDense);
q.setCutoutArgs(xrange,yrange,zrange,1);
im = oo.query(q);

% Visualize results
h = image(im); h.associate(anno);


%% get ac3


%%
coordMcell = [3.06510e+03 4.83910e+03 3.51550e+03] * 1e-2; % in um
voxelcoord = ceil(coordMcell ./ conf.voxelsize);
cutoutcoord = voxelcoord - anno.xyzOffset;
anno.data(cutoutcoord(1), cutoutcoord(2), cutoutcoord(3))


%% correction of anno.data

% fix voxels that are not connected by the object by a face (with imopen?)

% fix voxels that are connected by only 1 face?

%%% select all voxels of the object


labelmask = D == 862;
nnz(labelmask)
% lm = bwmorph(labelmask, 'bridge');
% lm = imclose(labelmask, se);
lm = imdilate(labelmask, se);
nnz(lm)
clear no el
[no,el] = binsurface(lm, 3);
[no, el] = meshcheckrepair(no, el, 'deep');
for ii = 1:3
    no(:,ii) = no(:,ii) - conf.voxelsize(ii); % correct for padding
    no(:,ii) = no(:,ii) * conf.voxelsize(ii) + anno.xyzOffset(ii) * conf.voxelsize(ii);
end
savebinstl(no, el, [datadir filesep conf.cu filesep 'test.stl'], 'test');


'bridge'
'close'
'diag'
'fill'
'majority'
'open'
'spur'
