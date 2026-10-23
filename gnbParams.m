function P = gnbParams(X, y, cls)
% STEP 6 (fit): Gaussian Naive Bayes parameters.
% For every feature and every class, estimate the mean and standard deviation
% of that feature among the training samples of the class.
%   P.mu(c,j), P.sd(c,j)  class c in {1=REAL, 2=FAKE}, feature j
%   P.prior               equal priors (the training data is SMOTE-balanced)

P.mu = zeros(numel(cls), size(X,2));
P.sd = zeros(numel(cls), size(X,2));
for c = 1:numel(cls)
    Xc = X(y == cls{c}, :);
    P.mu(c,:) = mean(Xc, 1);
    P.sd(c,:) = std(Xc, 0, 1) + 1e-6;
end
P.prior = [0.5 0.5];
end
