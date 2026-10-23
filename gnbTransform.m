function Z = gnbTransform(X, P, mode)
% STEP 6 (apply): turn MFCC features into a probabilistic representation using
% the class-conditional Gaussians learned by gnbParams.
%
%   mode 'posterior'  : Z(i,j) = P(FAKE | x_ij), the probability of the FAKE class
%                       given feature j alone (Bayes rule, equal priors).
%                       Output [n x d], every value in (0,1), non-negative.
%   mode 'likelihood' : [N(x_ij | REAL) , N(x_ij | FAKE)] for every feature.
%                       Output [n x 2d], non-negative.
%
% Both outputs are non-negative, as required by NMF in Step 7.

d = size(X, 2);
logL = zeros(size(X,1), d, 2);
for c = 1:2
    logL(:,:,c) = -0.5 * ((X - P.mu(c,:)) ./ P.sd(c,:)).^2 - log(P.sd(c,:)) - 0.5*log(2*pi);
end

switch char(mode)
    case 'posterior'
        a = logL(:,:,2) + log(P.prior(2));       % log P(x|FAKE) P(FAKE)
        b = logL(:,:,1) + log(P.prior(1));       % log P(x|REAL) P(REAL)
        Z = 1 ./ (1 + exp(b - a));
    case 'likelihood'
        Z = [exp(logL(:,:,1)), exp(logL(:,:,2))];
    otherwise
        error('Unknown GNB mode: %s', char(mode));
end
end
