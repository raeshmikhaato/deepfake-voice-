function G = hpGrid(name)
% Hyperparameter candidates for each classifier (a struct array, one element
% per candidate). Used by Step 8 for tuning. Shrink these for the full dataset.

switch name
    case 'GNB'      % nothing to tune
        G = struct('Unused', 0);
    case 'LR'       % regularisation strength
        G = struct('LRLambda', num2cell([1e-4 1e-3 1e-2 1e-1]));
    case 'RF'       % number of trees x minimum leaf size
        [T, L] = ndgrid([50 100 200], [1 5 10]);
        G = struct('RFTrees', num2cell(T(:)'), 'RFLeaf', num2cell(L(:)'));
    case 'KNN'      % number of neighbours x distance weighting
        w = {'equal', 'inverse'};
        [K, W] = ndgrid([3 5 7 11 15], [1 2]);
        G = struct('KNNk', num2cell(K(:)'), 'KNNweight', w(W(:)'));
    otherwise
        error('Unknown classifier: %s', name);
end
end
