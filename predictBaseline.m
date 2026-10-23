function [lab, score] = predictBaseline(m, X, posClass)
% Applies a model from fitBaseline to new segments.
%   lab    predicted class labels
%   score  classifier score for posClass (used for ROC / AUC)

Z = (X - m.mu) ./ m.sd;
[lab, sc] = predict(m.clf, Z);
col   = find(strcmp(string(m.clf.ClassNames), posClass));
score = sc(:, col);
end
