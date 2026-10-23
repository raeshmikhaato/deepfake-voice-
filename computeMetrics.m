function r = computeMetrics(yTrue, yPred, score, posClass)
% Accuracy, precision, recall, F1 and AUC, with posClass as the positive class.
pos = yTrue == posClass;
pp  = yPred == posClass;
TP = sum(pos & pp);  FP = sum(~pos & pp);  FN = sum(pos & ~pp);
r.acc  = mean(yTrue == yPred);
r.prec = TP / max(TP + FP, 1);
r.rec  = TP / max(TP + FN, 1);
r.f1   = 2 * r.prec * r.rec / max(r.prec + r.rec, eps);
[~, ~, ~, r.auc] = perfcurve(yTrue, score, posClass);
end
