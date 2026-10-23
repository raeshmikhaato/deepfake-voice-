function f = segmentFeatures(seg, cfg)
% STEP 4 core: one 1-second segment -> one feature vector.
% MFCCs are computed on short frames (30 ms window, 20 ms overlap, the
% mfcc() defaults), then summarised over the segment by the mean and the
% standard deviation of each coefficient -> [1 x 2*numCoeffs].

c = mfcc(seg, cfg.fs, 'NumCoeffs', cfg.numCoeffs, 'LogEnergy', 'Ignore');   % [frames x numCoeffs]
f = [mean(c, 1), std(c, 0, 1)];
end
