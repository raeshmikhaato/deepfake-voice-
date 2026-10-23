function [Xb, yb] = smoteBalance(X, y, k)
% SMOTE: oversample the minority class(es) with synthetic samples created by
% interpolating between a minority sample and one of its k nearest minority
% neighbours.   X [n x d] features,  y categorical labels,  k neighbours.
% Returns the original rows first, followed by the synthetic ones.

cls  = categories(y);
n    = cellfun(@(c) sum(y == c), cls);
nMax = max(n);
Xb = X;  yb = y;
for i = 1:numel(cls)
    nNew = nMax - n(i);
    if nNew == 0, continue; end
    Xm = X(y == cls{i}, :);
    kk = min(k, size(Xm,1) - 1);
    if kk < 1, continue; end
    idx = knnsearch(Xm, Xm, 'K', kk + 1);
    idx(:,1) = [];                                          % drop the point itself
    r   = randi(size(Xm,1), nNew, 1);                       % base samples
    nb  = idx(sub2ind(size(idx), r, randi(kk, nNew, 1)));   % a random neighbour of each
    Xnew = Xm(r,:) + rand(nNew,1) .* (Xm(nb,:) - Xm(r,:));  % point on the line between them
    Xb = [Xb; Xnew];                                        %#ok<AGROW>
    yb = [yb; categorical(repmat(cls(i), nNew, 1), cls)];   %#ok<AGROW>
end
end
