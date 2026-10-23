%% STEP 1: DATASET
% Indexes every audio file (path, split, class), prints dataset statistics,
% plots example waveforms, and saves dataset_index.mat for the next steps.

clear; clc; close all; rng(42);
cfg = projectConfig();

%% 1. Index all files
ads = audioDatastore(cfg.dataRoot, 'IncludeSubfolders', true, ...
    'FileExtensions', {'.mp3','.wav','.flac'});
tok = regexp(ads.Files, '(training|testing|validation)[\\/](real|fake)[\\/]', 'tokens', 'once');
ok  = ~cellfun(@isempty, tok);
tok = vertcat(tok{ok});
T = table(string(ads.Files(ok)), string(tok(:,1)), upper(string(tok(:,2))), ...
    'VariableNames', {'File','Split','Class'});

fprintf('Total audio files: %d\n\n', height(T));

%% 2. Counts per split and class
disp(groupsummary(T, {'Split','Class'}));

%% 3. Duration and sampling rate (random sample of 200 files)
n   = min(200, height(T));
idx = randperm(height(T), n);
dur = zeros(n,1);  sr = zeros(n,1);  ch = zeros(n,1);
for i = 1:n
    info   = audioinfo(char(T.File(idx(i))));
    dur(i) = info.Duration;  sr(i) = info.SampleRate;  ch(i) = info.NumChannels;
end
fprintf('Duration (s): min %.2f | median %.2f | max %.2f\n', min(dur), median(dur), max(dur));
fprintf('Clips shorter than 1 s: %.1f%% of the sample\n', 100*mean(dur < 1));
fprintf('Sampling rates found: %s Hz\n', mat2str(unique(sr)'));
fprintf('Channel counts found: %s\n', mat2str(unique(ch)'));

%% 4. Example waveforms
r = find(T.Class == "REAL", 1);   f = find(T.Class == "FAKE", 1);
[xr, fsr] = audioread(char(T.File(r)));
[xf, fsf] = audioread(char(T.File(f)));
figure('Position', [100 100 800 450]);
subplot(2,1,1); plot((0:numel(xr)-1)/fsr, xr); title('REAL voice'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(2,1,2); plot((0:numel(xf)-1)/fsf, xf); title('FAKE voice'); xlabel('Time (s)'); ylabel('Amplitude');
exportgraphics(gcf, fullfile(cfg.figDir, 'step01_waveforms.png'), 'Resolution', 200);

%% 5. Save the index for Step 2
save('dataset_index.mat', 'T');
fprintf('\nSaved dataset_index.mat and figures/step01_waveforms.png\n');
