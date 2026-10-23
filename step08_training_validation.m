%% STEP 8: MODEL TRAINING AND VALIDATION
% Trains GNB, LR, RF and KNN on both feature paths:
%   baseline     original MFCC features                    (Step 5)
%   transformed  MFCC -> GNB -> NMF features               (Steps 6-7)
%
% (a) Hyperparameter tuning: train on the official 'training' split, score F1
%     on the official 'validation' split. The held-out 'testing' split is never
%     touched here (it is used once, in Step 9).
% (b) 10-fold cross-validation on training + validation (folds grouped by
%     recording, from Step 5) with the tuned settings.
% Output: step08_<tag>.mat (tuned settings, CV predictions and scores).

clear; clc; close all; rng(42);
cfg = projectConfig();
load(sprintf('features_%s.mat', cfg.tag), 'X', 'y', 'split');
load(sprintf('prep_%s.mat', cfg.tag), 'dev', 'segFold');
Xd = X(dev,:);  yd = y(dev);  sp = split(dev);
isTr = sp == "training";  isVa = sp == "validation";
models = cfg.models;  pos = cfg.posClass;

%% 8a-1. Tune the NMF rank for the transformed path (with KNN, which is fast)
ranks = [4 6 8 10 12 15];
f1r = zeros(size(ranks));
for i = 1:numel(ranks)
    c2 = cfg;  c2.nmfRank = ranks(i);
    m = fitTransformed(Xd(isTr,:), yd(isTr), 'KNN', c2);
    [lab, sc] = predictTransformed(m, Xd(isVa,:), pos);
    f1r(i) = computeMetrics(yd(isVa), lab, sc, pos).f1;
    fprintf('NMF rank %2d: validation F1 %.4f\n', ranks(i), f1r(i));
end
[~, bi] = max(f1r);  bestRank = ranks(bi);
cfgT = cfg;  cfgT.nmfRank = bestRank;
fprintf('--> NMF rank chosen: %d\n\n', bestRank);

figure('Position', [100 100 600 380]);
plot(ranks, f1r, '-o', 'LineWidth', 1.3); grid on; xline(bestRank, '--');
xlabel('NMF components'); ylabel('Validation F1 (KNN)'); title('Choosing the NMF rank');
exportgraphics(gcf, fullfile(cfg.figDir, 'step08_nmf_rank.png'), 'Resolution', 200);

%% 8a-2. Tune each classifier on both paths
paths = {'baseline', 'transformed'};
hpBest = struct();  tune = {};
for p = 1:2
    for c = 1:numel(models)
        G = hpGrid(models{c});  f1 = zeros(1, numel(G));
        for g = 1:numel(G)
            if p == 1
                m = fitBaseline(Xd(isTr,:), yd(isTr), models{c}, cfg, G(g));
                [lab, sc] = predictBaseline(m, Xd(isVa,:), pos);
            else
                m = fitTransformed(Xd(isTr,:), yd(isTr), models{c}, cfgT, G(g));
                [lab, sc] = predictTransformed(m, Xd(isVa,:), pos);
            end
            f1(g) = computeMetrics(yd(isVa), lab, sc, pos).f1;
        end
        [bestF1, gi] = max(f1);
        hpBest.(paths{p}).(models{c}) = G(gi);
        tune(end+1,:) = {paths{p}, models{c}, jsonencode(G(gi)), bestF1}; %#ok<SAGROW>
        fprintf('%-12s %-4s best validation F1 %.4f  with %s\n', paths{p}, models{c}, bestF1, jsonencode(G(gi)));
    end
end
tuneTable = cell2table(tune, 'VariableNames', {'Path','Classifier','BestSettings','ValidationF1'});
writetable(tuneTable, sprintf('step08_tuning_%s.csv', cfg.tag));

%% 8b. 10-fold cross-validation with the tuned settings
fprintf('\n=== %d-fold cross-validation (tuned settings) ===\n', cfg.kFold);
N = numel(yd);  cv = struct();  rows = {};
for p = 1:2
    for c = 1:numel(models)
        hp = hpBest.(paths{p}).(models{c});
        pred = yd;  score = zeros(N,1);  tt = zeros(cfg.kFold,1);
        for k = 1:cfg.kFold
            te = segFold == k;
            if p == 1
                m = fitBaseline(Xd(~te,:), yd(~te), models{c}, cfg, hp);
                [pred(te), score(te)] = predictBaseline(m, Xd(te,:), pos);
            else
                m = fitTransformed(Xd(~te,:), yd(~te), models{c}, cfgT, hp);
                [pred(te), score(te)] = predictTransformed(m, Xd(te,:), pos);
            end
            tt(k) = m.trainTime;
        end
        cv.(paths{p}).(models{c}) = struct('pred', pred, 'score', score, 'trainTime', tt);
        r = computeMetrics(yd, pred, score, pos);
        rows(end+1,:) = {paths{p}, models{c}, r.acc, r.f1, r.auc, mean(tt)}; %#ok<SAGROW>
        fprintf('%-12s %-4s  Acc %.4f  F1 %.4f  AUC %.4f  (%.2fs per fold)\n', ...
            paths{p}, models{c}, r.acc, r.f1, r.auc, mean(tt));
    end
end
cvTable = cell2table(rows, 'VariableNames', {'Path','Classifier','Accuracy','F1','AUC','TrainTimeSec'});
fprintf('\n');  disp(cvTable);

save(sprintf('step08_%s.mat', cfg.tag), 'hpBest', 'bestRank', 'cfgT', 'cv', 'yd', 'tuneTable', 'cvTable');
fprintf('Saved step08_%s.mat. Step 9 turns these results into the full evaluation.\n', cfg.tag);
