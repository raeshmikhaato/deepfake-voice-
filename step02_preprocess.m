%% STEP 2: AUDIO PREPROCESSING
% Noise (rumble) removal, volume normalisation, one-second segmentation,
% filtering of corrupted / silent segments. The preprocessing itself lives in
% preprocessAudio.m so later steps reuse exactly the same code.
%
% This script: (a) picks the working subset, (b) checks the files' metadata
% for class bias, (c) shows before/after plots, (d) runs preprocessing over
% the subset and reports statistics. Output: subset_index.mat

clear; clc; close all; rng(42);
cfg = projectConfig();
load('dataset_index.mat', 'T');

%% 2a. Choose a balanced working subset (per split and class)
keep = false(height(T), 1);
for s = ["training","validation","testing"]
    for c = ["REAL","FAKE"]
        idx = find(T.Split == s & T.Class == c);
        idx = idx(randperm(numel(idx), min(numel(idx), cfg.maxPerClassPerSplit)));
        keep(idx) = true;
    end
end
Tsub = T(keep, :);
fprintf('Working subset: %d files\n\n', height(Tsub));

%% 2b. Metadata bias check: do real and fake differ in sampling rate / channels / length?
n = height(Tsub);
sr = zeros(n,1);  ch = zeros(n,1);  dur = zeros(n,1);
for i = 1:n
    info = audioinfo(char(Tsub.File(i)));
    sr(i) = info.SampleRate;  ch(i) = info.NumChannels;  dur(i) = info.Duration;
end
Tsub.SampleRate = sr;  Tsub.Channels = ch;  Tsub.Duration = dur;

disp('--- Sampling rate by class ---');
disp(groupsummary(Tsub, {'Class','SampleRate'}));
disp('--- Channels by class ---');
disp(groupsummary(Tsub, {'Class','Channels'}));
disp('--- Duration (s) by class ---');
disp(groupsummary(Tsub, 'Class', {'mean','median','max'}, 'Duration'));

%% 2c. Before / after example (one REAL, one FAKE)
figure('Position', [100 100 950 600]);
cls = ["REAL","FAKE"];
for k = 1:2
    i = find(Tsub.Class == cls(k), 1);
    [x, fsIn] = audioread(char(Tsub.File(i)));
    [segs, xp] = preprocessAudio(x, fsIn, cfg);
    subplot(2,2,2*k-1); plot((0:size(x,1)-1)/fsIn, mean(x,2));
    title(sprintf('%s: raw (%d Hz)', cls(k), fsIn)); xlabel('Time (s)'); ylim([-1 1]);
    subplot(2,2,2*k);   plot((0:numel(xp)-1)/cfg.fs, xp);
    title(sprintf('%s: preprocessed (%d Hz), %d segments', cls(k), cfg.fs, size(segs,2)));
    xlabel('Time (s)'); ylim([-1 1]);
end
exportgraphics(gcf, fullfile(cfg.figDir, 'step02_before_after.png'), 'Resolution', 200);

%% 2d. Run preprocessing over the whole subset and collect statistics
nSeg = zeros(n,1);  nQuietAll = zeros(n,1);  bad = false(n,1);  t0 = tic;
for i = 1:n
    try
        [x, fsIn] = audioread(char(Tsub.File(i)));
        [segs, ~, nq] = preprocessAudio(x, fsIn, cfg);
        nSeg(i) = size(segs, 2);  nQuietAll(i) = nq;
    catch
        bad(i) = true;                              % unreadable / corrupted file
    end
    if mod(i, 500) == 0, fprintf('  %d/%d files (%.1f min)\n', i, n, toc(t0)/60); end
end
Tsub.NSeg = nSeg;

fprintf('\nUnreadable files: %d\n', sum(bad));
fprintf('Files with no usable segment: %d\n', sum(nSeg == 0 & ~bad));
fprintf('Silent segments removed: %d\n', sum(nQuietAll));
fprintf('Total usable 1-s segments: %d\n\n', sum(nSeg));
disp(groupsummary(Tsub, {'Split','Class'}, 'sum', 'NSeg'));

figure; histogram(Tsub.NSeg(Tsub.Class=="REAL"), 'BinMethod', 'integers'); hold on
histogram(Tsub.NSeg(Tsub.Class=="FAKE"), 'BinMethod', 'integers');
legend('REAL','FAKE'); xlabel('Usable 1-s segments per file'); ylabel('Files');
title('Segments per recording');
exportgraphics(gcf, fullfile(cfg.figDir, 'step02_segments_per_file.png'), 'Resolution', 200);

%% Save for Step 3 onward
Tsub = Tsub(~bad & Tsub.NSeg > 0, :);
save('subset_index.mat', 'Tsub');
fprintf('Saved subset_index.mat (%d usable files)\n', height(Tsub));
