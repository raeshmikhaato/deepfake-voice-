%% STEP 3: DATASET BALANCING USING SMOTE
% (a) measure the class imbalance at SEGMENT level (what the classifiers see)
% (b) show how capping segments per recording changes it
% (c) test smoteBalance.m on a toy 2-D dataset and plot before / after
%
% SMOTE needs feature vectors, so it is applied to the MFCC features (Step 4)
% inside each training fold in Step 8; it is never applied to test data.

clear; clc; close all; rng(42);
cfg = projectConfig();
load('subset_index.mat', 'Tsub');

%% 3a. Segment-level balance from Step 2
fprintf('Files in subset: %d\n\n', height(Tsub));
G = groupsummary(Tsub, {'Split','Class'}, 'sum', 'NSeg');
disp(G);
nReal = sum(Tsub.NSeg(Tsub.Class == "REAL"));
nFake = sum(Tsub.NSeg(Tsub.Class == "FAKE"));
fprintf('Total segments  REAL: %d   FAKE: %d   ratio REAL:FAKE = %.2f\n\n', nReal, nFake, nReal/nFake);

%% 3b. Effect of capping the number of segments taken from one recording
caps = [1 2 3 Inf];
R = zeros(numel(caps), 4);
for j = 1:numel(caps)
    s = min(Tsub.NSeg, caps(j));
    nR = sum(s(Tsub.Class == "REAL"));  nF = sum(s(Tsub.Class == "FAKE"));
    R(j,:) = [caps(j) nR nF nR/nF];
end
disp(array2table(R, 'VariableNames', {'MaxSegPerFile','RealSegments','FakeSegments','RealToFakeRatio'}));
fprintf('Chosen cap in projectConfig.m: cfg.maxSegPerFile = %d\n\n', cfg.maxSegPerFile);

%% 3c. SMOTE demo on a toy 2-D dataset (checks smoteBalance.m works)
nMaj = 400;  nMin = 100;
Xmaj = randn(nMaj, 2) * 0.8;
Xmin = randn(nMin, 2) * 0.5 + [2.5 2];
X = [Xmaj; Xmin];
y = categorical([repmat("REAL", nMaj, 1); repmat("FAKE", nMin, 1)], cfg.classes);

[Xb, yb] = smoteBalance(X, y, cfg.smoteK);
fprintf('Toy data before SMOTE: REAL %d, FAKE %d\n', sum(y=="REAL"),  sum(y=="FAKE"));
fprintf('Toy data after  SMOTE: REAL %d, FAKE %d\n', sum(yb=="REAL"), sum(yb=="FAKE"));
assert(sum(yb=="REAL") == sum(yb=="FAKE"), 'SMOTE did not balance the classes');

nOrig = size(X, 1);
isSyn = false(size(Xb,1), 1);  isSyn(nOrig+1:end) = true;

figure('Position', [100 100 1000 420]);
subplot(1,2,1); hold on; grid on
scatter(X(y=="REAL",1), X(y=="REAL",2), 12, 'filled');
scatter(X(y=="FAKE",1), X(y=="FAKE",2), 12, 'filled');
title('Before SMOTE'); legend('REAL (majority)', 'FAKE (minority)'); xlabel('feature 1'); ylabel('feature 2');
subplot(1,2,2); hold on; grid on
scatter(Xb(yb=="REAL" & ~isSyn,1), Xb(yb=="REAL" & ~isSyn,2), 12, 'filled');
scatter(Xb(yb=="FAKE" & ~isSyn,1), Xb(yb=="FAKE" & ~isSyn,2), 12, 'filled');
scatter(Xb(isSyn,1), Xb(isSyn,2), 18, 'x');
title('After SMOTE'); legend('REAL', 'FAKE (original)', 'FAKE (synthetic)'); xlabel('feature 1');
exportgraphics(gcf, fullfile(cfg.figDir, 'step03_smote_demo.png'), 'Resolution', 200);
fprintf('\nSaved figures/step03_smote_demo.png\n');
