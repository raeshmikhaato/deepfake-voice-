%% STEP 6: GNB-BASED FEATURE PROCESSING
% Gaussian Naive Bayes turns every MFCC feature into a probabilistic feature:
% for each feature we learn one Gaussian per class (REAL, FAKE), then express
% each value through those Gaussians.
%   (a) plot the fitted Gaussians against the real data for three features
%   (b) show how the 'posterior' representation separates the classes
%   (c) compare the representations with a quick 10-fold CV (KNN and RF),
%       to decide which one Step 7 (NMF) should receive.
% All fitting inside (c) uses the training part of each fold only.

clear; clc; close all; rng(42);
cfg = projectConfig();
load(sprintf('features_%s.mat', cfg.tag), 'X', 'y', 'featNames');
load(sprintf('prep_%s.mat', cfg.tag), 'dev', 'segFold');
Xd = X(dev,:);  yd = y(dev);
cls = cfg.classes;
cR = [0 0.447 0.741];  cF = [0.85 0.325 0.098];

%% 6a. The Gaussians GNB learns (fitted on all development data, for illustration)
P = gnbParams(Xd, yd, cls);
show = [find(featNames == "mfcc0_std"), find(featNames == "mfcc5_mean"), find(featNames == "mfcc1_mean")];
figure('Position', [100 100 1200 380]);
tiledlayout(1, 3, 'TileSpacing', 'compact');
for s = show
    nexttile; hold on; grid on
    histogram(Xd(yd=="REAL", s), 30, 'Normalization', 'pdf', 'FaceColor', cR, 'FaceAlpha', 0.35, 'EdgeColor', 'none');
    histogram(Xd(yd=="FAKE", s), 30, 'Normalization', 'pdf', 'FaceColor', cF, 'FaceAlpha', 0.35, 'EdgeColor', 'none');
    xs = linspace(min(Xd(:,s)), max(Xd(:,s)), 300);
    plot(xs, normpdf(xs, P.mu(1,s), P.sd(1,s)), 'Color', cR, 'LineWidth', 1.8);
    plot(xs, normpdf(xs, P.mu(2,s), P.sd(2,s)), 'Color', cF, 'LineWidth', 1.8);
    title(featNames(s), 'Interpreter', 'none'); xlabel('Feature value'); ylabel('Density');
    if s == show(1), legend('REAL data','FAKE data','REAL Gaussian','FAKE Gaussian'); end
end
sgtitle('GNB models each feature with one Gaussian per class');
exportgraphics(gcf, fullfile(cfg.figDir, 'step06_gaussian_fits.png'), 'Resolution', 200);

%% 6b. Probabilistic representations
Zpost = gnbTransform(Xd, P, 'posterior');       % [n x 26], P(FAKE | feature j)
Zlik  = gnbTransform(Xd, P, 'likelihood');      % [n x 52]
fprintf('posterior  representation: %d x %d, range [%.3f, %.3f]\n', size(Zpost,1), size(Zpost,2), min(Zpost(:)), max(Zpost(:)));
fprintf('likelihood representation: %d x %d, range [%.4f, %.3f]\n\n', size(Zlik,1), size(Zlik,2), min(Zlik(:)), max(Zlik(:)));

figure('Position', [100 100 950 420]);
mR = mean(Zpost(yd=="REAL",:), 1);  mF = mean(Zpost(yd=="FAKE",:), 1);
bar(1:numel(mR), [mR; mF]'); grid on
xticks(1:numel(mR)); xticklabels(featNames); xtickangle(60); ylim([0 1]);
ylabel('Mean P(FAKE | feature)'); legend('REAL segments', 'FAKE segments');
title('Posterior features: REAL segments sit low, FAKE segments sit high');
exportgraphics(gcf, fullfile(cfg.figDir, 'step06_posterior_by_class.png'), 'Resolution', 200);

%% 6c. Which representation keeps the information? (quick CV, in-fold fitting)
modes   = ["none", "posterior", "likelihood"];
labels  = ["Standardised MFCC (baseline)", "GNB posterior (26)", "GNB likelihood (52)"];
clfs    = {'KNN', 'RF'};
rows = {};
for m = 1:numel(modes)
    for c = 1:numel(clfs)
        r = cvRepresentation(Xd, yd, segFold, cfg, modes(m), clfs{c});
        rows(end+1,:) = {char(labels(m)), clfs{c}, r.acc, r.f1, r.auc}; %#ok<SAGROW>
        fprintf('%-30s %-4s  Acc %.4f  F1 %.4f  AUC %.4f\n', labels(m), clfs{c}, r.acc, r.f1, r.auc);
    end
end
cmp = cell2table(rows, 'VariableNames', {'Representation','Classifier','Accuracy','F1','AUC'});
fprintf('\n');  disp(cmp);
writetable(cmp, sprintf('step06_representation_check_%s.csv', cfg.tag));

%% ---------------------------------------------------------------------
function r = cvRepresentation(Xd, yd, segFold, cfg, mode, clfName)
% 10-fold CV of one representation. Standardisation, SMOTE and GNB fitting all
% use the training part of the fold only.
N = numel(yd);  pred = yd;  score = zeros(N,1);
for k = 1:cfg.kFold
    te  = segFold == k;
    mu  = mean(Xd(~te,:), 1);  sd = std(Xd(~te,:), 0, 1) + eps;
    Ztr = (Xd(~te,:) - mu) ./ sd;
    Zte = (Xd(te,:)  - mu) ./ sd;
    [Ztr, ytr] = smoteBalance(Ztr, yd(~te), cfg.smoteK);
    if mode ~= "none"
        P = gnbParams(Ztr, ytr, cfg.classes);
        A = gnbTransform(Ztr, P, mode);
        B = gnbTransform(Zte, P, mode);
        if mode == "likelihood"                     % likelihoods are unbounded: scale to [0,1]
            zmax = max(A, [], 1) + eps;
            A = min(A ./ zmax, 1);  B = min(B ./ zmax, 1);
        end
        Ztr = A;  Zte = B;
    end
    clf = trainClassifier(clfName, Ztr, ytr);
    [pred(te), sc] = predict(clf, Zte);
    score(te) = sc(:, strcmp(string(clf.ClassNames), cfg.posClass));
end
r = computeMetrics(yd, pred, score, cfg.posClass);
end
