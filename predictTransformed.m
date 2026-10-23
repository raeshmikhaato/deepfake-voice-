function [lab, score] = predictTransformed(m, X, posClass)
% Applies a model from fitTransformed to new segments.
%   lab    predicted class labels
%   score  classifier score for posClass (used for ROC / AUC)

Z = (X - m.mu) ./ m.sd;
A = gnbTransform(Z, m.P, m.mode);
if ~isempty(m.zmax), A = min(A ./ m.zmax, 1); end
W = nmfProject(A, m.H);
[lab, sc] = predict(m.clf, (W - m.wmu) ./ m.wsd);
score = sc(:, strcmp(string(m.clf.ClassNames), posClass));
end
