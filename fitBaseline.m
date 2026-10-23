function m = fitBaseline(X, y, clfName, cfg, hp)
% Baseline path: original MFCC features -> standardise -> SMOTE -> classifier.
% Everything is learned from the training data passed in (one CV fold's
% training part), so no information leaks from the test fold.
%
% Standardisation comes BEFORE SMOTE on purpose: SMOTE picks neighbours by
% distance, and without standardisation c0 (range of hundreds) would decide
% every neighbour choice.

if nargin < 5, hp = struct(); end
t0 = tic;
m.mu = mean(X, 1);
m.sd = std(X, 0, 1) + eps;
Z = (X - m.mu) ./ m.sd;
[Z, y] = smoteBalance(Z, y, cfg.smoteK);
m.clf = trainClassifier(clfName, Z, y, hp);
m.trainTime = toc(t0);
end
