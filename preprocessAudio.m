function [segs, xp, nQuiet] = preprocessAudio(x, fsIn, cfg)
% STEP 2 core: raw audio -> cleaned, normalised 1-second segments.
%   x      raw samples (any channels), fsIn its sampling rate
%   segs   [samplesPerSegment x nSegments] cleaned segments
%   xp     the full processed signal (for plotting)
%   nQuiet number of near-silent segments that were removed

L = round(cfg.fs * cfg.segSec);
H = round(cfg.fs * cfg.hopSec);
segs = zeros(L, 0);  xp = [];  nQuiet = 0;

% Filter out corrupted audio
if isempty(x) || any(~isfinite(x(:))), return; end

x = mean(x, 2);                                   % stereo -> mono
if fsIn ~= cfg.fs
    x = resample(x, cfg.fs, fsIn);                % common sampling rate
end
x = filter([1 -1], [1 -0.995], x);                % remove DC offset / low-frequency rumble

peak = max(abs(x));
if peak == 0, return; end
x = x / peak;                                     % volume normalisation (peak = 1)
xp = x;

if numel(x) < L, return; end                      % clip shorter than one segment
starts = 1:H:(numel(x) - L + 1);
segs = x(starts + (0:L-1)');                      % one-second segmentation

quiet = sqrt(mean(segs.^2, 1)) < cfg.minRMS;      % filter near-silent segments
nQuiet = sum(quiet);
segs(:, quiet) = [];
end
