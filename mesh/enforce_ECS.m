function [L, Bb, Lb, Nb, Ab, rp, idxs] = enforce_ECS(L)
%%% create connected ECS mask

% load the labelimage
% [L, Lclass] = get_labelimage(conf);

% initialize ECS mask
M = L==0;

% % remove border components (NOT BW = imclearborder(BW, 4));
% borderlabels = [];
% borderlabels = cat(1, borderlabels, unique(L(1:end,1)));
% borderlabels = cat(1, borderlabels, unique(L(1:end,end)));
% borderlabels = cat(1, borderlabels, unique(L(1,1:end))');
% borderlabels = cat(1, borderlabels, unique(L(end,1:end))');
% for i = 1:length(borderlabels)
%     M(L == borderlabels(i)) = true;
% end

% null components not in contact with ECS (to avoid inducing ECS witin fibres)
for c = conf.conn.ECSfree
    L(Lclass==c) = 0;
end

% for every pixel neighbours should be either same label or '0'
connc = defconn(conf.conn.ECS);
for i = 1:size(connc, 1)
    K = circshift(L, [connc(i,1), connc(i,2)]);
    M = M | (L ~= K & (L ~= 0 & K ~= 0));
end

%%% get the boundaries of the ECS-nulled labelimage
BW = ~M;
% introduce the nested components (alternating labels/nulled)
for c = conf.conn.nested
    BW(Lclass==c) = false;
end

[Bb,Lb,Nb,Ab] = bwboundaries(BW, conf.conn.B);

idxs.all = 1:length(Bb);
idxs.parents = 1:Nb;
idxs.children = Nb+1:length(Bb);
rp = regionprops(Lb, Lclass, 'MeanIntensity');
for i = 1:length(conf.classes)
    idxs.(conf.classes{i}) = find([rp.MeanIntensity]==i);
end
