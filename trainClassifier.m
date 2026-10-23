function clf = trainClassifier(name, F, y, hp)
% Trains one of the four classifiers used in the project.
%   name  'GNB' | 'LR' | 'RF' | 'KNN'
%   F     [n x d] features,  y  categorical labels
%   hp    optional struct of hyperparameters (fields missing -> defaults):
%           LRLambda, RFTrees, RFLeaf, KNNk, KNNweight

if nargin < 4 || isempty(hp), hp = struct(); end
switch name
    case 'GNB'      % Gaussian Naive Bayes
        clf = fitcnb(F, y);
    case 'LR'       % Logistic Regression (ridge-regularised)
        clf = fitclinear(F, y, 'Learner', 'logistic', 'Regularization', 'ridge', ...
            'Lambda', opt(hp, 'LRLambda', 'auto'));
    case 'RF'       % Random Forest (bagged decision trees)
        clf = fitcensemble(F, y, 'Method', 'Bag', ...
            'NumLearningCycles', opt(hp, 'RFTrees', 100), ...
            'Learners', templateTree('MinLeafSize', opt(hp, 'RFLeaf', 1)));
    case 'KNN'      % K-Nearest Neighbours
        clf = fitcknn(F, y, 'NumNeighbors', opt(hp, 'KNNk', 5), ...
            'DistanceWeight', opt(hp, 'KNNweight', 'equal'));
    otherwise
        error('Unknown classifier: %s', name);
end
end

function v = opt(hp, field, default)
if isfield(hp, field), v = hp.(field); else, v = default; end
end
