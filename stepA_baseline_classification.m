%% BRANCH A: BASELINE CLASSIFICATION (ORIGINAL MFCC FEATURES)
% Trains GNB, LR, RF and KNN directly on the original MFCC features with
% 10-fold cross-validation (folds grouped by recording, from Step 5).
% Runs twice: with all 26 features, and without the two c0 features, to show
% how much the loudness-related c0 contributes.
% Output: baseline_<tag>.mat, baseline_results_<tag>.csv and report figures.

clear; clc; close all; rng(42);
cfg = projectConfig();
load(sprintf('features_%s.mat', cfg.tag), 'X', 'y');
load(sprintf('prep_%s.mat', cfg.tag), 'dev', 'segFold');
Xd = X(dev,:);  yd = y(dev);  N = numel(yd);

nAll = size(X, 2);
variants = struct( ...
    'name', {'All 26 MFCC features', 'Without c0 (24 features)'}, ...
    'cols', {1:nAll, setdiff(1:nAll, [1, cfg.numCoeffs + 1])});

rows = {};  store = struct();
for v = 1:numel(variants)
    cols = variants(v).cols;
    fprintf('\n=== %s ===\n', variants(v).name);
    for c = 1:numel(cfg.models)
        pred = yd;  score = zeros(N,1);  tt = zeros(cfg.kFold,1);
        for k = 1:cfg.kFold
            te = segFold == k;
            m  = fitBaseline(Xd(~te,cols), yd(~te), cfg.models{c}, cfg);
            [pred(te), score(te)] = predictBaseline(m, Xd(te,cols), cfg.posClass);
            tt(k) = m.trainTime;
        end
        r = computeMetrics(yd, pred, score, cfg.posClass);
        rows(end+1,:) = {variants(v).name, cfg.models{c}, r.acc, r.prec, r.rec, r.f1, r.auc, mean(tt)}; %#ok<SAGROW>
        store(v).(cfg.models{c}) = struct('pred', pred, 'score', score);
        fprintf('%-4s  Acc %.4f  Prec %.4f  Rec %.4f  F1 %.4f  AUC %.4f  (%.2fs per fold)\n', ...
            cfg.models{c}, r.acc, r.prec, r.rec, r.f1, r.auc, mean(tt));
    end
end

results = cell2table(rows, 'VariableNames', ...
    {'Features','Classifier','Accuracy','Precision','Recall','F1','AUC','TrainTimeSec'});
fprintf('\n');  disp(results);
writetable(results, sprintf('baseline_results_%s.csv', cfg.tag));
save(sprintf('baseline_%s.mat', cfg.tag), 'results', 'store', 'variants', 'yd');

%% Figure 1: ROC curves (all 26 features)
figure('Position', [100 100 650 550]); hold on; grid on
for c = 1:numel(cfg.models)
    [fpr, tpr, ~, auc] = perfcurve(yd, store(1).(cfg.models{c}).score, cfg.posClass);
    plot(fpr, tpr, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (AUC %.3f)', cfg.models{c}, auc));
end
plot([0 1], [0 1], 'k--', 'HandleVisibility', 'off');
xlabel('False positive rate'); ylabel('True positive rate');
title('Baseline MFCC features: ROC curves (10-fold CV)'); legend('Location', 'southeast');
exportgraphics(gcf, fullfile(cfg.figDir, 'stepA_roc.png'), 'Resolution', 200);

%% Figure 2: confusion matrices (all 26 features)
figure('Position', [100 100 800 700]);
tiledlayout(2, 2, 'TileSpacing', 'compact');
for c = 1:numel(cfg.models)
    nexttile;
    confusionchart(yd, store(1).(cfg.models{c}).pred, 'Title', cfg.models{c});
end
sgtitle('Baseline MFCC features: confusion matrices');
exportgraphics(gcf, fullfile(cfg.figDir, 'stepA_confusion.png'), 'Resolution', 200);

%% Figure 3: effect of removing c0 (F1 score)
f1All = results.F1(strcmp(results.Features, variants(1).name));
f1No0 = results.F1(strcmp(results.Features, variants(2).name));
figure('Position', [100 100 650 420]);
bar(categorical(cfg.models, cfg.models), [f1All, f1No0]); grid on
ylabel('F1 score'); ylim([0 1]); legend(variants(1).name, variants(2).name, 'Location', 'southeast');
title('Baseline: with and without the c0 features');
exportgraphics(gcf, fullfile(cfg.figDir, 'stepA_c0_ablation.png'), 'Resolution', 200);
