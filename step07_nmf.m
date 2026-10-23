%% STEP 7: NMF FEATURE TRANSFORMATION
% Non-negative Matrix Factorization compresses the GNB probabilistic features
% (matrix A, segments x features) into a few non-negative components:
%       A  ~  W * H
%   H  [components x features]  the "building blocks" (basis)
%   W  [segments x components]  how much of each block every segment contains;
%                               W is the new feature set given to the classifiers.
%
% (a) build the NMF input exactly as the pipeline does   (b) reconstruction
% error against the number of components   (c) the learned basis H
% (d) how well each new component separates REAL from FAKE.
% Fitted on all development data here, for illustration only; Step 8 refits
% inside every cross-validation fold.

clear; clc; close all; rng(42);
cfg = projectConfig();
load(sprintf('features_%s.mat', cfg.tag), 'X', 'y', 'featNames');
load(sprintf('prep_%s.mat', cfg.tag), 'dev');
Xd = X(dev,:);  yd = y(dev);

%% 7a. NMF input = standardise -> SMOTE -> GNB representation (Steps 3 and 6)
mu = mean(Xd, 1);  sd = std(Xd, 0, 1) + eps;
Z  = (Xd - mu) ./ sd;
[Zb, yb] = smoteBalance(Z, yd, cfg.smoteK);
P  = gnbParams(Zb, yb, cfg.classes);
A    = gnbTransform(Zb, P, cfg.gnbMode);
Adev = gnbTransform(Z,  P, cfg.gnbMode);
if strcmp(cfg.gnbMode, 'likelihood')
    zmax = max(A, [], 1) + eps;
    A = min(A ./ zmax, 1);  Adev = min(Adev ./ zmax, 1);
    colNames = ["REAL: " + featNames, "FAKE: " + featNames];
else
    colNames = featNames;
end
d = size(A, 2);
fprintf('NMF input (%s mode): %d segments x %d features\n\n', cfg.gnbMode, size(A,1), d);

%% 7b. Reconstruction error versus number of components
ranks = 2:min(30, d-1);
err = zeros(size(ranks));
for i = 1:numel(ranks)
    [W, H] = nnmf(A, ranks(i), 'replicates', 2, 'options', statset('MaxIter', 200));
    err(i) = norm(A - W*H, 'fro') / norm(A, 'fro');
end
figure('Position', [100 100 700 420]);
plot(ranks, err, '-o', 'LineWidth', 1.3); grid on; hold on
xline(cfg.nmfRank, '--', sprintf('rank = %d', cfg.nmfRank));
xlabel('Number of NMF components'); ylabel('Relative reconstruction error');
title('NMF: how much information is kept as components are added');
exportgraphics(gcf, fullfile(cfg.figDir, 'step07_reconstruction_error.png'), 'Resolution', 200);

%% 7c. Final factorisation at the configured rank
k = min(cfg.nmfRank, d);
[Wtr, H] = nnmf(A, k, 'replicates', 3, 'options', statset('MaxIter', 300));
fprintf('Chosen rank %d: relative reconstruction error %.3f\n', k, norm(A - Wtr*H, 'fro') / norm(A, 'fro'));
Wdev = nmfProject(Adev, H);                      % new features of the real (non-synthetic) segments
fprintf('New feature matrix: %d segments x %d components\n\n', size(Wdev,1), size(Wdev,2));

figure('Position', [100 100 1000 480]);
imagesc(H); colorbar; axis xy
xticks(1:d); xticklabels(colNames); xtickangle(90); set(gca, 'TickLabelInterpreter', 'none');
ylabel('NMF component'); title('NMF basis H: which input features each component is built from');
exportgraphics(gcf, fullfile(cfg.figDir, 'step07_basis.png'), 'Resolution', 200);

%% 7d. How well does each new component separate REAL from FAKE? (AUC, direction-free)
isF = yd == "FAKE";  n1 = sum(isF);  n0 = sum(~isF);
aucC = zeros(1, k);
for j = 1:k
    r = tiedrank(Wdev(:,j));
    aucC(j) = (sum(r(isF)) - n1*(n1+1)/2) / (n1*n0);
end
aucC = max(aucC, 1 - aucC);
[aucSorted, order] = sort(aucC, 'descend');
fprintf('Best single NMF components (AUC):\n');
disp(table(order(1:5)', aucSorted(1:5)', 'VariableNames', {'Component','AUC'}));

figure('Position', [100 100 800 400]);
bar(aucSorted); grid on; ylim([0.5 1]);
xticks(1:k); xticklabels(string(order)); xlabel('NMF component (sorted)'); ylabel('Single-component AUC');
title('Separation power of the NMF components');
exportgraphics(gcf, fullfile(cfg.figDir, 'step07_component_auc.png'), 'Resolution', 200);

figure('Position', [100 100 650 520]);
gscatter(Wdev(:,order(1)), Wdev(:,order(2)), yd, [], [], 8); grid on
xlabel(sprintf('NMF component %d', order(1))); ylabel(sprintf('NMF component %d', order(2)));
title('The two most separating NMF components');
exportgraphics(gcf, fullfile(cfg.figDir, 'step07_scatter.png'), 'Resolution', 200);
