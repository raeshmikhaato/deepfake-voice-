%% STEP 4: MFCC FEATURE EXTRACTION
% For every file in the working subset: preprocess (Step 2), keep at most
% cfg.maxSegPerFile evenly spaced segments (Step 3), compute MFCC features for
% each segment. Output: features_<tag>.mat with X (features), y (labels),
% fileId, split, featNames. Also saves two report figures.

clear; clc; close all; rng(42);
cfg = projectConfig();
load('subset_index.mat', 'Tsub');
nFiles = height(Tsub);
nFeat  = 2 * cfg.numCoeffs;

%% 4a. Extract features file by file
Xc = cell(nFiles,1); yc = cell(nFiles,1); idc = cell(nFiles,1); sc = cell(nFiles,1);
nBad = 0;  t0 = tic;
for i = 1:nFiles
    try
        [x, fsIn] = audioread(char(Tsub.File(i)));
        segs = preprocessAudio(x, fsIn, cfg);
        nS = size(segs, 2);
        if nS == 0, continue; end
        if nS > cfg.maxSegPerFile                       % cap, evenly spaced through the clip
            segs = segs(:, round(linspace(1, nS, cfg.maxSegPerFile)));
        end
        F = zeros(size(segs,2), nFeat);
        for s = 1:size(segs,2)
            F(s,:) = segmentFeatures(segs(:,s), cfg);
        end
        F = F(all(isfinite(F), 2), :);
        n = size(F, 1);
        if n == 0, continue; end
        Xc{i} = F;
        yc{i} = repmat(Tsub.Class(i), n, 1);
        idc{i} = repmat(i, n, 1);
        sc{i} = repmat(Tsub.Split(i), n, 1);
    catch ME
        nBad = nBad + 1;
        warning('Skipped %s (%s)', Tsub.File(i), ME.message);
    end
    if mod(i, 250) == 0
        fprintf('  %d/%d files (%.1f min)\n', i, nFiles, toc(t0)/60);
    end
end
keep   = ~cellfun(@isempty, Xc);
X      = vertcat(Xc{keep});
y      = categorical(vertcat(yc{keep}), cfg.classes);
fileId = vertcat(idc{keep});
split  = vertcat(sc{keep});
k = 0:cfg.numCoeffs-1;
featNames = [compose("mfcc%d_mean", k), compose("mfcc%d_std", k)];

fprintf('\nFeature matrix: %d segments x %d features  (%d files failed)\n', size(X,1), size(X,2), nBad);
disp(groupsummary(table(split, y, 'VariableNames', {'Split','Class'}), {'Split','Class'}));
save(sprintf('features_%s.mat', cfg.tag), 'X', 'y', 'fileId', 'split', 'featNames');
fprintf('Saved features_%s.mat\n', cfg.tag);

%% 4b. Figure 1: MFCCs of one REAL and one FAKE segment
figure('Position', [100 100 950 420]);
cls = ["REAL","FAKE"];
for j = 1:2
    i = find(Tsub.Class == cls(j), 1);
    [x, fsIn] = audioread(char(Tsub.File(i)));
    segs = preprocessAudio(x, fsIn, cfg);
    c = mfcc(segs(:,1), cfg.fs, 'NumCoeffs', cfg.numCoeffs, 'LogEnergy', 'Ignore');
    subplot(1,2,j); imagesc(c'); axis xy; colorbar;
    title(sprintf('%s: MFCCs of one 1-s segment', cls(j)));
    xlabel('Frame'); ylabel('MFCC index');
end
exportgraphics(gcf, fullfile(cfg.figDir, 'step04_mfcc_examples.png'), 'Resolution', 200);

%% 4c. Figure 2: average MFCC profile per class (mean features, +/- 1 std)
figure('Position', [100 100 800 420]); hold on; grid on
for j = 1:2
    Xj = X(y == cls(j), 1:cfg.numCoeffs);
    errorbar(k + (j-1.5)*0.2, mean(Xj,1), std(Xj,0,1), '-o', 'LineWidth', 1.2);
end
legend('REAL','FAKE'); xlabel('MFCC index'); ylabel('Mean value over a segment');
title('Average MFCC profile by class');
exportgraphics(gcf, fullfile(cfg.figDir, 'step04_mfcc_class_profile.png'), 'Resolution', 200);
