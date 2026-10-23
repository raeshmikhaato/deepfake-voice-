function W = nmfProject(Z, H)
% STEP 7 helper: express new samples in an existing NMF basis.
% Given the basis H learned on the training data (Z ~ W*H), find the
% non-negative coefficients W for the rows of Z with H held fixed
% (multiplicative updates). Deterministic: starts from the clipped
% least-squares solution.

W = max(Z / H, 0) + 1e-3;
HHt = H * H';
ZHt = Z * H';
for it = 1:200
    W = W .* ZHt ./ (W * HHt + eps);
end
end
