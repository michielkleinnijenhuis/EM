function [no, el] = shaped_boundary_3DEM(mask, mu_margin, upsamplefactor, pxsize, method, opt)

mask_up = imresize(mask, upsamplefactor);
pxsize_up = pxsize / upsamplefactor; % in um
se_width_px = ceil(repmat(mu_margin, 3,1)./pxsize_up');
se = ones(se_width_px(1),se_width_px(2),se_width_px(3)); % box
% se = strel3d(se_width_px(1)); % ball
% se = strel('ball', se_width_px(1), se_width_px(1), 2); % is non-flat

% expand the mask to accomodate the dilated image
mask_pad = padarray(mask_up, se_width_px);

% dilate
mask_dil = imdilate(mask_pad, se);

% find connected components
CC = bwconncomp(mask_dil);

% create mask for the largest component
numPixels = cellfun(@numel, CC.PixelIdxList);
[~,I] = max(numPixels);
M = false(size(mask_dil));
M(CC.PixelIdxList{I}) = true;

% get boundary of the largest component
[no, el, ~, ~]=vol2surf(M, 1:size(M,1), 1:size(M,2), 1:size(M,3), ...
    opt, 1, method, 1);

[no, el] = meshcheckrepair(no, el, 'deep');
for ii = 1:3
    no(:,ii) = no(:,ii) - se_width_px(ii); % correct for padding
end
no = no * 1/upsamplefactor; % correct for imresize
