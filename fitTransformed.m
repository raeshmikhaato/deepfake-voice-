function m = fitTransformed(X, y, clfName, cfg, hp)
% Transformed path (Steps 6-7):
%   MFCC -> standardise -> SMOTE -> GNB probabilistic features -> NMF -> classifier
% Everything is learned from the training data passed in, so no information
% leaks from the test fold.

if nargin < 5, hp = struct(); end
t0 = tic;
m.mu = mean(X, 1);
m.sd = std(X, 0, 1) + eps;
Z = (X - m.mu) ./ m.sd;
[Z, y] = smoteBalance(Z, y, cfg.smoteK);

% Step 6: GNB probabilistic representation
m.mode = cfg.gnbMode;
m.P = gnbParams(Z, y, cfg.classes);
A = gnbTransform(Z, m.P, m.mode);
m.zmax = [];
if strcmp(m.mode, 'likelihood')                    % unbounded -> scale to [0,1]
    m.zmax = max(A, [], 1) + eps;
    A = min(A ./ m.zmax, 1);
end

% Step 7: NMF
rank = min(cfg.nmfRank, size(A, 2));
[~, m.H] = nnmf(A, rank, 'replicates', 3, 'options', statset('MaxIter', 300));
W = nmfProject(A, m.H);
m.wmu = mean(W, 1);
m.wsd = std(W, 0, 1) + eps;

m.clf = trainClassifier(clfName, (W - m.wmu) ./ m.wsd, y, hp);
m.trainTime = toc(t0);
end
