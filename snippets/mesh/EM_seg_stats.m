clear all; clc;

% datadir = '/Users/mkleinnijenhuis/Dropbox/stats_ISMRM2017';
% comp = {'stats_MF'; 'stats_MA'; 'stats_UA';};
% names = {'myelinated fibre'; 'myelinated axon'; 'unmyelinated axon'};
% es = 0.0511;

datadir = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blender/stats';
comp = {'B-NT-S10-2f_ROI_00us_labels_labelMA_agglo-labelMAx0.5_l27200_u00000_s00010_filledm5'; ...
    'B-NT-S10-2f_ROI_00_labels_labelMF-labelMF'; 'B-NT-S10-2f_ROI_00'};
names = {'myelinated axon'; 'myelinated fibre'};
es = 0.007;

fs = 28;
col1 = [0, 0.45, 0.74];
col2 = [0.85, 0.33, 0.1];
% col1 = [0, 1, 1];
% col2 = [1, 0, 1];

meas = 'AD';
% meas = 'eccentricity';

slcws_MA = dlmread([datadir filesep comp{1} '_slcws_' meas '.txt']);
slcws_MF = dlmread([datadir filesep comp{2} '_slcws_' meas '.txt']);
% slcws_UA = dlmread([datadir filesep comp{3} '_slcws_AD.txt']);
slcws_GR = dlmread([datadir filesep comp{3} '_gratios_slcws.txt']);
gratios = dlmread([datadir filesep comp{3} '_gratios_gratios.txt']);
slcws_TH = (slcws_MF - slcws_MA) / 2;

mask = ones(size(slcws_MA));
mask(isnan(slcws_MA)) = 0;  % 541216
mask(isnan(slcws_MF)) = 0;  % 545153
% 545153 - 541216 = 3937
mask(isnan(slcws_GR)) = 0;  % 541216
mask(slcws_GR>0.99) = 0;  % 616995
% 184*3605 - 616995 = 46325
gratios = gratios(~isnan(gratios));

% % mask the first 10 and final 3 sections
mask(1:10, :) = 0;
mask(181:184, :) = 0;

% % mask where MA does not exist in all sections (N=1389)
% invalid_axon_mask = sum(isnan(slcws_MA)) ~= 0;
% mask(:,invalid_axon_mask) = 0;

ADs = {slcws_MA; slcws_MF; slcws_GR};
% ADs = {slcws_MA; slcws_MF; slcws_UA; slcws_GR};




%% plotting AD profiles over z

% AD = th_(11:end-4, :); clim = [0 pi];
% AD = phi_(11:end-4, :); clim = [0 pi/2];
AD{1} = slcws_MA(11:end-4, :) * es; clim = [0 3];
AD{2} = slcws_MF(11:end-4, :) * es; clim = [0 3];
AD{3} = slcws_TH(11:end-4, :) * es; clim = [0 0.3];
AD{4} = slcws_GR(11:end-4, :); clim = [0 1];
AD{5} = deg2rad(deg(11:end-4, :)); clim = [0 pi/2];
clims = [0 2; 0 2; 0 0.3; 0.3 0.9; 0 pi/2];
titles = {'inner', 'outer', 'thickness', 'g-ratio', 'angle'}

% AD = slcws_MA(11:end-4, :) * es;
maskFULLaxons = ~isnan(sum(AD{1}));
naxons = sum(maskFULLaxons);

sel = randperm(naxons, 1000);

%
% figure, plotbrowser,

% ADfilt = zeros(170, naxons);
% ADfilt = AD(:, maskFULLaxons);

% plot(ADfilt * es);

% figure
% a = linspace(1, sum(maskFULLaxons), 36);
% b = a + 36;
% for i = 1:36
%     subplot(6,6,i)
%     plot(ADfilt(:,a(i):b(i)) * es);
% end

xl = [0 200];
xt = [];
yl = [0 170];
yt = [];

for j = 1:5

    ax = subplot(5,1,j);
    ax.Position = [0.1000 1-(j)*0.19 0.8 0.17];
    ADfilt = zeros(170, naxons);
    ADfilt = AD{j}(:, maskFULLaxons);
    i = 1;

%     ax = subplot(5,1,i)

    ADf = ADfilt(:, sel(i*200-199:i*200));
    imagesc(ADf);

    % % sorted version (confusing)
    % ADfilt = sort(ADfilt);
    % [B,I] = sort(mean(ADfilt));
    % imagesc(ADfilt(:,fliplr(I)));

    % % interleave with white bar (not pretty)
    % A = ones(size(ADfilt)) * 3;
    % C = ADfilt(:,[1;1]*(1:size(ADfilt,2)));
    % C(:,1:2:end) = A;
    % imagesc(C);

    xlim(xl);
    xticks(xt);
    ylim(yl);
    yticks(yt);

    set(gca,'fontsize', fs);
    ylabel(titles{j});
    ax.YAxisLocation = 'right';
    colormap hot
%     colormap cool
    axis xy
    caxis(clims(j,:))
%     caxis(clim)
    colorbar
    if j < 5
%         axis off
    end

end


%%
Y = peak2peak(ADfilt);
figure, hist(Y);
for axon = 1:100:naxons
%     [pks,locs] = findpeaks(ADfilt(:,axon), 'MinPeakDistance',10,
%     'MinPeakHeight',0.5);
    [pks,locs] = findpeaks(ADfilt(:,axon), 'MinPeakDistance', 10, 'MinPeakProminence', 0.1);
    plot(1:170, ADfilt(:,axon),locs,pks,'o'); hold on;
end
axon = 6;
[pks,locs] = findpeaks(ADfilt(:,axon), 'MinPeakDistance', 10, 'MinPeakProminence', 0.1);
plot(1:170, ADfilt(:,axon),locs,pks,'o'); hold on;

%%
cv_ax = std(ADfilt, '', 1, 'omitnan') ./ mean(ADfilt, 'omitnan');
% figure, hist(cv_ax, 100);
h = histfit(cv_ax, 30, 'gamma');

%% AD histograms
figure,
xl = [0 2];
xt = [0,0.5, 1, 1.5, 2];
yl = [0 18000];
yt = [0 9000 18000];
nbins = [200, 200, 200];
% yl = [0 30000];
% yt = [0 15000 30000];

for i = 1:2 %length(comp)

    AD = ADs{i}(mask==1);
%     AD(AD<1.8) = [];  %0.0999 %0.1153
    AD = AD * es;
    subplot(2,3,i);
    h = histfit(AD, nbins(i), 'gamma'); hold on;
    h = histfit(AD, nbins(i), 'gev');
    set(gca,'fontsize', fs);
    set(h(1),'FaceColor', col1);
    set(h(1),'EdgeColor', col1);
    xlim(xl);
    xticks(xt);
    ylim(yl);
    yticks(yt);
    title([names{i} ' diameter'], 'fontsize', fs);
    axis square;
    mean(AD(:), 'omitnan')
    median(AD(:), 'omitnan')
    std(AD(:), 'omitnan')
%     p = gamfit(AD(:))
%     p = gevfit(AD(:))
end


%% random permutation

AD = ADs{1};  % NOTE: needs to be nslices*axons
met = AD*es;
met(mask==0) = NaN;
name = 'axon diameter';

rp = met(randperm(numel(met)));
rp = reshape(rp, size(met));

var_ax = var(met, '', 1, 'omitnan');
cv_ax = std(met, '', 1, 'omitnan') ./ mean(met, 'omitnan');

var_rp = var(rp, '', 1, 'omitnan');
cv_rp = std(rp, '', 1, 'omitnan') ./ mean(rp, 'omitnan');

ax = subplot(2,3,3);

bins = linspace(0.0, 0.25, 100);
xl = [0, 0.2];
xt = [0, 0.05, 0.1, 0.15, 0.2];
yl = [0, 300];
yt = [0, 150, 300];
h1 = histogram(ax, var_ax(~isnan(var_ax)), bins); hold on;
h2 = histogram(ax, var_rp(~isnan(var_rp)), bins);

xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
set(ax,'fontsize', fs); axis square;
title([name ' variance'], 'fontsize', fs);
h1.EdgeColor = col1;
h2.EdgeColor = col2;
h1.FaceColor = col1;
h2.FaceColor = col2;


%% CV

AD= ADs{1};
met = AD * es;
met(mask==0) = NaN;
cv_inner = std(met, '', 1, 'omitnan') ./ mean(met, 'omitnan');

AD = ADs{2};
met = AD * es;
met(mask==0) = NaN;
cv_outer = std(met, '', 1, 'omitnan') ./ mean(met, 'omitnan');

ax = subplot(2,3,6);
bins = linspace(0.0, 1.0, 100);
xl = [0, 1];
xt = [0, 1];
yl = [0, 300];
yt = [0, 300];
h1 = histogram(ax, cv_inner(~isnan(cv_inner)), bins); hold on;
h2 = histogram(ax, cv_outer(~isnan(cv_outer)), bins);

xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
set(ax,'fontsize', fs); axis square;
title([name ' variance'], 'fontsize', fs);
h1.EdgeColor = col1;
h2.EdgeColor = col2;
h1.FaceColor = col1;
h2.FaceColor = col2;


%%
% gratios calculated from areas or read from file
% TODO: take out ones
calcg = 1;
if calcg
    xl = [0 1];
%     areasMM = dlmread([datadir filesep comp{1} '_slcws_area.txt']);
%     areaMM = sum(areasMM, 'omitnan');
    meas = 'area';
    areasMA = dlmread([datadir filesep comp{1} '_slcws_' meas '.txt']);
    areasMF = dlmread([datadir filesep comp{2} '_slcws_' meas '.txt']);
    areasMM = areasMF - areasMA;
    areaMA = sum(areasMA, 'omitnan');
    areaMF = sum(areasMF, 'omitnan');
    areaMM = sum(areasMM, 'omitnan');
    grat = sqrt(1 - areaMM ./ areaMF);  % NOTE: not same mask as slcws
    grat_slcws = sqrt(1 - areasMM ./ areasMF);
else
    grat = dlmread([datadir filesep 'B-NT-S10-2f_ROI_00_gratios_gratios.txt']);
    grat_slcws = dlmread([datadir filesep 'B-NT-S10-2f_ROI_00_gratios_slcws.txt']);
end

%% slicewise g-ratio

GR = slcws_GR(mask==1);  % GR = (slcws_MA(mask==1) / 2) ./ (slcws_MF(mask==1) / 2); % GR = grat_slcws(mask==1); OK: SAME
xl = [0, 1];
xt = [0, 0.25, 0.50, 0.75, 1.0];
yl = [0, 10000];
yt = [0, 5000, 10000];
subplot(2,3,4);
% h = histfit(GR, 200, 'gamma');
hist(GR, 200)
xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
set(gca,'fontsize', fs);
h = findobj(gca,'Type','patch');
set(h,'FaceColor', col1);
set(h,'EdgeColor', col1);
xlim(xl); axis square;
title('slicewise g-ratio', 'fontsize', fs);
p = gamfit(GR(:))
mean(GR(:), 'omitnan')
median(GR(:), 'omitnan')
std(GR(:), 'omitnan')

% h = histfit(GR, 200, 'gamma');
% h = histfit(GR, 200, 'normal');
% h = histfit(GR, 200, 'logistic');
% h = histfit(GR, 200, 'tlocationscale');

%% axon g-ratio from slicewise

xl = [0, 1];
xt = [0, 0.25, 0.50, 0.75, 1.0];
yl = [0, 120];
yt = [0, 60, 120];
subplot(2,3,5);

%     areasMM = dlmread([datadir filesep comp{1} '_slcws_area.txt']);
%     areaMM = sum(areasMM, 'omitnan');
%     areasMA = dlmread([datadir filesep comp{2} '_slcws_area.txt']);
%     areaMA = sum(areasMA, 'omitnan');
%     areaML = areaMM - areaMA; plot([areaMA; areaMM; areaML]')
%     gratios = sqrt(1 - areaML ./ areaMM);

MM = areasMM; MM(mask==0) = NaN;
MA = areasMA; MA(mask==0) = NaN;
MF = areasMF; MF(mask==0) = NaN;
MM_area_tot = sum(MM, 'omitnan');
MA_area_tot = sum(MA, 'omitnan');
MF_area_tot = sum(MF, 'omitnan');
grat = sqrt(1 - MM_area_tot ./ MF_area_tot);
grat(grat>0.99) = [];
hist(grat, 200)

mean(grat(:), 'omitnan')
median(grat(:), 'omitnan')
std(grat(:), 'omitnan')

xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
h = findobj(gca,'Type','patch');
set(h,'FaceColor', col1);
set(h,'EdgeColor', col1);
set(gca,'fontsize', fs);
xlim(xl); axis square;
title('g-ratio', 'fontsize', fs);

%% axon g-ratio from seg_stats.py

subplot(2,3,5);
% gratios = gratios(~isnan(gratios));
gratios(gratios>0.99) = [];
% h = histfit(gratios, 200, 'gamma');
% h = histfit(gratios, 200, 'normal');
% h = histfit(gratios, 200, 'logistic');
% h = histfit(gratios, 200, 'tlocationscale');
hist(gratios, 200)
xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
h = findobj(gca,'Type','patch');
set(h,'FaceColor', col1);
set(h,'EdgeColor', col1);
set(gca,'fontsize', fs);
xlim(xl); axis square;
title('g-ratio', 'fontsize', fs);

mean(gratios(:), 'omitnan')
median(gratios(:), 'omitnan')
std(gratios(:), 'omitnan')

%%

ax = subplot(2,3,3);

h1 = histogram(ax, grat(~isnan(grat)), bins); hold on;
h2 = histogram(ax, gratios(~isnan(gratios)), bins);
xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
set(ax,'fontsize', fs); axis square;
title([name ' variance'], 'fontsize', fs);
h1.EdgeColor = col1;
h2.EdgeColor = col2;
h1.FaceColor = col1;
h2.FaceColor = col2;




%% random permutation

met = slcws_GR;
met(mask==0) = NaN;
% met = gratios_slcws;

name = 'g-ratio';

sd_ax = var(met, '', 1, 'omitnan');
rp = met(randperm(numel(met)));
rp = reshape(rp, size(met));
sd_rp = var(rp, '', 1, 'omitnan');

bins = linspace(0.0, 0.08, 100);
xl = [0, 0.04];
xt = [0, 0.01, 0.02, 0.03, 0.04];
yl = [0, 600];
yt = [0, 300, 600];

ax = subplot(2,3,5);
h1 = histogram(ax, sd_ax(~isnan(sd_ax)), bins); hold on;
h2 = histogram(ax, sd_rp(~isnan(sd_rp)), bins);
xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
set(ax,'fontsize', fs); axis square;
title([name ' variance'], 'fontsize', fs);
h1.EdgeColor = col1;
h2.EdgeColor = col2;
h1.FaceColor = col1;
h2.FaceColor = col2;


%% CV g-ratio

met = slcws_GR;
met(mask==0) = NaN;
cv_inner = std(met, '', 1, 'omitnan') ./ mean(met, 'omitnan');

ax = subplot(2,3,6);
bins = linspace(0.0, 1.0, 100);
xl = [0, 0.5];
xt = [0, 0.25, 0.5];
yl = [0, 300];
yt = [0, 300];
h1 = histogram(ax, cv_inner(~isnan(cv_inner)), bins); hold on;

xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
set(ax,'fontsize', fs); axis square;
title([name ' CoV'], 'fontsize', fs);
h1.EdgeColor = col1;
h2.EdgeColor = col2;
h1.FaceColor = col1;
h2.FaceColor = col2;


%%
fig1 = figure(1);
% ax1 = subplot(2,3,1);
ax1 = axes;

fac = 2;
nbins=512;

MA = slcws_MA(mask==1) * es;
GR = slcws_GR(mask==1);
b = polyfit(log(MA), GR, 1);      % ln y = mx + b
x = fac * logspace(-2, 0, nbins);

yfit = b(1) * log(x) + b(2);

xl = [x(1), x(end)];
xt = [x(1), fac/10, x(end)];
yl = [0, 1];
yt = [0, 0.25, 0.5, 0.75, 1];


xedges = x
yedges = linspace(0, 1, nbins);
[N,XEDGES,YEDGES] = histcounts2(GR,MA,yedges,xedges);
[X, Y] = meshgrid(xedges(2:end), yedges(2:end));
h = pcolor(X, Y, N);  % rot90(N,1)
h.EdgeColor = 'none'
set(gca,'fontsize', fs, 'XScale', 'log'); axis square;
colormap cool

hold on;
grid on;

% scatter(MA, GR, 'Marker', '.', ...
%     'MarkerFaceAlpha', 0.01, 'MarkerEdgeAlpha', 0.01, ...
%     'MarkerFaceColor', [0, 0.45, 0.74], 'MarkerFaceColor', [0, 0.45, 0.74]);

% points = [MA, GR];
% grid = 1024;   %refinement of map
% % minvals = min(points);
% % maxvals = max(points);
% minvals = [0.02, 0];
% maxvals = [2, 1];
% rangevals = maxvals - minvals;
% xidx = 1 + round((points(:,1) - minvals(1)) ./ rangevals(1) * (grid-1));
% yidx = 1 + round((points(:,2) - minvals(2)) ./ rangevals(2) * (grid-1));
% density = accumarray([yidx, xidx], 1, [grid,grid]);  %note y is rows, x is cols
% % imagesc(density, 'xdata', [minvals(1), maxvals(1)], 'ydata', [minvals(2), maxvals(2)]);


axis square
xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);

plot(x, yfit, 'k-', 'LineWidth', 2);

title(['g-ratio vs AD'], 'fontsize', fs);
set(gca,'fontsize', fs, 'XScale', 'log'); axis square;


% ax2 = subplot(2,3,1);
ax2 = axes('Color','none');
linkaxes([ax1 ax2]);
axis square;
grid on;
ax2.GridLineStyle = ':'
ax2.GridColor = 'w'
ax2.GridAlpha = 1
xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);
set(ax2,'fontsize', fs, 'XScale', 'log');
set(ax2,'XTickLabel',[]);
set(ax2,'YTickLabel',[]);
set(ax2,'xcol','w','ycol','w')
% tx1 = text(50,50,txt1,'units','data','backgroundcolor',[1 0 0])


%% mean over slices
axon_mMA = zeros(1, 3605);
axon_mGR = zeros(1, 3605);
for axon = 1:size(slcws_MA, 2)
    axon_mask = mask(:, axon);

    axon_MA = slcws_MA(:, axon) * es;
    MA = axon_MA(axon_mask==1);
    axon_mMA(axon) = mean(MA, 'omitnan');

    % note gratios not present for all axons: there are some fails
    axon_GR = slcws_GR(:, axon);
    GR = axon_GR(axon_mask==1);
    axon_mGR(axon) = mean(GR, 'omitnan');
end

% ind = 1:length(a); 
% k = axon_nMA(~isnan(axon_mMA)); p = polyfit(ind(k),a(k),n)

b = polyfit(log(axon_mMA(~isnan(axon_mMA))), axon_mGR(~isnan(axon_mGR)), 1)      % ln y = mx + b
x = linspace(0.001, 3, 300);
yfit = b(1) * log(x) + b(2);

xl = [0.02, 2];
xt = [0.02, 0.2, 2];
yl = [0, 1];
yt = [0, 0.25, 0.5, 0.75, 1];

subplot(2,3,4);
scatter(axon_mMA, axon_mGR, 'Marker', '.', ...
    'MarkerFaceAlpha', 0.01, 'MarkerEdgeAlpha', 0.01, ...
    'MarkerFaceColor', [0, 0.45, 0.74], 'MarkerFaceColor', [0, 0.45, 0.74]);
axis square
xlim(xl);
xticks(xt);
ylim(yl);
yticks(yt);

hold on;
plot(x, yfit, 'r-');


% slcws_AD = dlmread([datadir filesep comp{1} '_slcws_AD.txt']);
% meanAD = mean(slcws_AD*es, 1);
% subplot(2,3,1); plot(meanAD', gratios, 'Linestyle', 'none', 'Marker', 'o')
title(['g-ratio vs AD'], 'fontsize', fs);
set(gca,'fontsize', fs, 'XScale', 'log'); axis square;

%%
% slcws_CX = dlmread([datadir filesep 'slcws_centroid_x.txt']);
% slcws_CY = dlmread([datadir filesep 'slcws_centroid_y.txt']);
% skel = cat(3, slcws_CX', slcws_CY');
% plot(squeeze(skel(4,:,1)), squeeze(skel(4,:,2)));
% 

% mean(slcws_AD(:)*es, 'omitnan'); % 0.1827
% 
% % r = sqrt(A/pi);
% for a=1:4
% d = 2*sqrt(a/pi)
% end
% % * 0.0511
% 
% r = slcws_AD;
% hist(r(:), 100)
% mean(r(:)*es, 'omitnan'); % 0.1827
% histfit(r(:), 100, 'gamma')
% p = gamfit(r(:)); % 2.0822    0.0878

