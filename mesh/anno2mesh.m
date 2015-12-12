function [smalllabels, labelmask_all] = anno2mesh(stldir, annodata, labels, method, opt, voxelsize, cutoutoffset, nvoxelthreshold, conn)

dims = size(annodata);

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

switch conn
    case 26
        se = ones([3,3,3]); % 26-conn
    case 18
        se(:,:,1) = [0 1 0;1,1,1;0,1,0];
        se(:,:,2) = [1 1 1;1,1,1;1,1,1];
        se(:,:,3) = [0 1 0;1,1,1;0,1,0]; % 18-conn
    case 6
        se(:,:,1) = [0 0 0;0,1,0;0,0,0];
        se(:,:,2) = [0 1 0;1,1,1;0,1,0];
        se(:,:,3) = [0 0 0;0,1,0;0,0,0]; % 6-conn
end

se26 = ones([3,3,3]); % 26-conn
se18(:,:,1) = [0 1 0;1,1,1;0,1,0];
se18(:,:,2) = [1 1 1;1,1,1;1,1,1];
se18(:,:,3) = [0 1 0;1,1,1;0,1,0]; % 18-conn
se06(:,:,1) = [0 0 0;0,1,0;0,0,0];
se06(:,:,2) = [0 1 0;1,1,1;0,1,0];
se06(:,:,3) = [0 0 0;0,1,0;0,0,0]; % 6-conn

% se = ones([4,4,4]); % 26-conn

% upsamplefactor = 3;
padsize = 3;
D = padarray(annodata, [padsize,padsize,padsize]);
labelmask_all = false(size(D));

smalllabels = [];


for label = labels'

    clear no el

    labelmask = D == label;

    if nnz(labelmask) > nvoxelthreshold

        switch method
            case 'binsurf'
                % 12- and 26- connected voxels should be removed (e.g. imdilate with 26-conn se)
                % or dilate and remove voxels overlapping with other objects
%                 labelmask = imopen(labelmask, se); % leaves holes
%                 labelmask = imclose(labelmask, se); % results in non-manifolds (edge-conn 'holes')
%                 labelmask = imdilate(labelmask, se); % causes intersections
%                 labelmask = imresize(labelmask, upsamplefactor); labelmask = imdilate(labelmask, se); % still has intersections and non-manifold geometry
%                 for i = 1:8  % this isn't foolproof either, as it can introduce new edge-connected voxels (iterative?)
%                     labelmask(bwhitmiss(labelmask,inter(:,:,:,i))) = 0;
%                 end % still edge-conn (101, 117, 862, 2064, 3208, 3271) % still ridges/points without thickness (862)
%                 for i = 1:size(inter,4)  % this isn't foolproof either, as it can introduce new edge-connected voxels (iterative?)
%                     labelmask(bwhitmiss(labelmask,inter(:,:,:,i))) = 0;
%                     %TODO? alternatively fill the corner voxel
%                 end % this is starting to look good, but there are elements without thickness (imclose?)
                % also holes induces by taking out ridges with no thickness
%                 labelmask = imclose(labelmask, se); % not the desired effect, but reintroduces ridges (fills hole)
                % tophat, bothat? BW2= imfill(BW,locations,conn)?
%                 labelmask = labelmask | imbothat(labelmask, se); % might even do the complete trick withouth the hitmiss??? NOPE
%                 for i = 1:size(inter,4)  % this isn't foolproof either, as it can introduce new edge-connected voxels (iterative?)
%                     labelmask(bwhitmiss(~labelmask,inter(:,:,:,i))) = 1;
%                 end % this is starting to look good, but there are nodes without thickness
%                 labelmask = labelmask | imtophat(~labelmask, se); % mweh
%                 l1 = imdilate(labelmask, se);
%                 l2 = l1 & ~labelmask;
%                 l3 = imerode(l2, se06);
%                 labelmask = l2 - l3;
                
                labelmask = ~(imdilate(imerode(~labelmask, se26), se06));
                labelmask = labelmask(padsize+1:end-padsize,padsize+1:end-padsize,padsize+1:end-padsize);
                labelmask = padarray(labelmask, [padsize,padsize,padsize]);
                for i = 1:size(inter,4)
                    labelmask(bwhitmiss(~labelmask,inter(:,:,:,i))) = 1;
                end
                % remove any intersections
                labelmask(labelmask_all) = false;
                % check manifoldness
                labelmask_all(labelmask) = true;
                [no,el] = binsurface(labelmask, 3);
            case {'cgalsurf', 'simplify'}
                [no, el, ~, ~]=vol2surf(labelmask, 1:dims(1), 1:dims(2), 1:dims(3), ...
                    opt, 0, method, 1);
            case 'isosurf'
                % remove non-face connected voxels
%                 labelmask = imopen(labelmask, se06);
                labelmask = imclose(labelmask, se26);
%                 for i = 1:size(inter,4)
%                     labelmask(bwhitmiss(~labelmask,inter(:,:,:,i))) = 1;
%                 end
                fv = isosurface(labelmask,0.5);
                no = fv.vertices;
                el = fv.faces;
        end

        [no, el] = meshcheckrepair(no, el, 'deep');

        for ii = 1:3
%             if strcmp(method, 'binsurfaces')
%                 no(:,ii) = no(:,ii)  * 1/upsamplefactor; % correct for upsampling
%             end
            no(:,ii) = no(:,ii) - padsize * voxelsize(ii); % correct for padding
            no(:,ii) = no(:,ii) * voxelsize(ii) + ...
                cutoutoffset(ii) * voxelsize(ii); % convert to mu and translate to offset
        end
        savebinstl(no, el, [stldir filesep method filesep ...
            'U_A.' num2str(label, '%05d') '.01.stl'], num2str(label));
    else
        smalllabels = [smalllabels label];
    end
end

labelmask_all = labelmask_all(padsize+1:end-padsize,padsize+1:end-padsize,padsize+1:end-padsize);
