%% STEP 5: FEATURE PREPARATION (BASELINE INPUT)
% Prepares the original MFCC features for classification:
%  (a) splits them into development (training + validation) and held-out test sets
%  (b) checks the features: ranges, per-feature class separability, PCA view
%  (c) builds the 10-fold cross-validation folds, grouped by recording, and
%      saves them so every later step uses exactly the same folds.
% Output: prep_<tag>.mat

clear; clc; close all; rng(42);
cfg = projectConfig();
load(sprintf('features_%s.mat', cfg.tag), 'X', 'y', 'fileId', 'split', 'featNames');

%% 5a. Development / held-out split (official FoR splits)
dev = split ~= "testing";           % training + validation -> cross-validation
tst = split == "testing";           % never used for training or tuning
Xd = X(dev,:);  yd = y(dev);  idd = fileId(dev);
Xt = X(tst,:);  yt = y(tst);
fprintf('Development set: %d segments (REAL %d, FAKE %d)\n', size(Xd,1), sum(yd=="REAL"), sum(yd=="FAKE"));
fprintf('Held-out test  : %d segments (REAL %d, FAKE %d)\n', size(Xt,1), sum(yt=="REAL"), sum(yt=="FAKE"));
fprintf('Features per segment: %d   NaN/Inf values: %d\n\n', size(X,2), sum(~isfinite(X(:))));

%% 5b. Feature ranges (this is why standardisation is needed)
T = table(featNames(:), min(Xd)', mean(Xd)', max(Xd)', std(Xd)', ...
    'VariableNames', {'Feature','Min','Mean','Max','Std'});
disp(T(1:cfg.numCoeffs, :));   % the 13 "mean" features

%% 5c. Single-feature class separability (AUC, direction-free)
isF = yd == "FAKE";  n1 = sum(isF);  n0 = sum(~isF);
sep = zeros(1, size(Xd,2));
for j = 1:size(Xd,2)
    r = tiedrank(Xd(:,j));
    sep(j) = (sum(r(isF)) - n1*(n1+1)/2) / (n1*n0);
end
sep = max(sep, 1 - sep);                        % 0.5 = no separation, 1 = perfect
[sepSorted, order] = sort(sep, 'descend');
fprintf('Most separating single features (AUC):\n');
disp(table(featNames(order(1:8))', sepSorted(1:8)', 'VariableNames', {'Feature','AUC'}));

figure('Position', [100 100 950 420]);
bar(sepSorted); grid on
xticks(1:numel(order)); xticklabels(featNames(order)); xtickangle(60);
ylabel('Single-feature AUC'); ylim([0.5 1]); title('How well each MFCC feature separates REAL from FAKE');
exportgraphics(gcf, fullfile(cfg.figDir, 'step05_feature_separability.png'), 'Resolution', 200);

%% 5d. Class profile again, without c0 (so the other coefficients are visible)
k = 1:cfg.numCoeffs-1;                          % c1..c12
figure('Position', [100 100 800 420]); hold on; grid on
for j = ["REAL","FAKE"]
    Xj = Xd(yd == j, 1 + k);
    errorbar(k + 0.1*(j=="FAKE") - 0.05, mean(Xj,1), std(Xj,0,1), '-o', 'LineWidth', 1.2);
end
legend('REAL','FAKE'); xlabel('MFCC index (c0 excluded)'); ylabel('Mean value over a segment');
title('Average MFCC profile by class, without c0');
exportgraphics(gcf, fullfile(cfg.figDir, 'step05_profile_without_c0.png'), 'Resolution', 200);

%% 5e. PCA view of the standardised features
Xz = zscore(Xd);
[~, score, ~, ~, expl] = pca(Xz, 'NumComponents', 2);
figure('Position', [100 100 700 520]);
gscatter(score(:,1), score(:,2), yd, [], [], 8); grid on
xlabel(sprintf('PC1 (%.1f%% of variance)', expl(1)));
ylabel(sprintf('PC2 (%.1f%% of variance)', expl(2)));
title('Baseline MFCC features, first two principal components');
exportgraphics(gcf, fullfile(cfg.figDir, 'step05_pca.png'), 'Resolution', 200);

%% 5f. Cross-validation folds, grouped by recording
% All segments of one recording stay in the same fold, so the same speaker
% clip never appears in both training and test folds.
[ufile, ia] = unique(idd);
cvf = cvpartition(yd(ia), 'KFold', cfg.kFold);
foldOfFile = zeros(max(ufile), 1);
for f = 1:cfg.kFold
    foldOfFile(ufile(test(cvf, f))) = f;
end
segFold = foldOfFile(idd);                      % fold number of every development segment
fprintf('\nSegments per fold (REAL / FAKE):\n');
for f = 1:cfg.kFold
    fprintf('  fold %2d: %4d / %4d\n', f, sum(segFold==f & yd=="REAL"), sum(segFold==f & yd=="FAKE"));
end

save(sprintf('prep_%s.mat', cfg.tag), 'dev', 'tst', 'segFold');
fprintf('\nSaved prep_%s.mat\n', cfg.tag);
