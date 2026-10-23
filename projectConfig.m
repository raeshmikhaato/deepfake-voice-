function cfg = projectConfig()
% Shared settings for every step of the deepfake voice detection project.
% Keep this file in the same folder as the step scripts and the dataset.

cfg.root     = fileparts(mfilename('fullpath'));
cfg.dataRoot = fullfile(cfg.root, 'for-original');   % contains training/, validation/, testing/
cfg.figDir   = fullfile(cfg.root, 'figures');
if ~isfolder(cfg.figDir), mkdir(cfg.figDir); end

% Step 2: preprocessing
cfg.fs        = 16000;     % working sampling rate (Hz); lowest rate found in the dataset
cfg.segSec    = 1;         % segment length (s)
cfg.hopSec    = 1;         % hop between segments (s); 1 = non-overlapping
cfg.minRMS    = 0.01;      % drop near-silent segments after normalisation

% Step 3: balancing
cfg.maxSegPerFile = 2;     % cap on segments taken from one recording (used in Step 4)
cfg.smoteK        = 5;     % SMOTE neighbours

% Step 4: MFCC
cfg.numCoeffs = 13;

% Step 6: GNB probabilistic features
%   'posterior'  : P(FAKE | feature j) for each feature  -> 26 features in (0,1)
%   'likelihood' : N(x_j | REAL), N(x_j | FAKE) per feature -> 52 features
cfg.gnbMode   = 'posterior';

% Step 7: NMF
cfg.nmfRank   = 10;

% Step 8: validation
cfg.kFold     = 10;

% Quick-run switch: files per class per split (Inf = use all of them)
cfg.maxPerClassPerSplit = 300;
if isinf(cfg.maxPerClassPerSplit), cfg.tag = 'full'; else, cfg.tag = sprintf('sub%d', cfg.maxPerClassPerSplit); end

cfg.classes   = {'REAL','FAKE'};
cfg.posClass  = 'FAKE';
cfg.models    = {'GNB','LR','RF','KNN'};
end
