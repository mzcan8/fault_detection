myDir = uigetdir;
featureCount = 50;
maxEpochs = 200;
wsize = 100;
wshift = 40;

outarr = [5 9 12 17];
[inputData1,inputData2,targets1,targets2] = prepareBilstmData(myDir, wsize, wshift, outarr);

instanceCount1 = length(inputData1);

pTrain = 0.80 ;
pVal = 0.10;
pTest = 0.10 ;
idx = randperm(instanceCount1);
%idx = (1:instanceCount);
trainInd = idx(1:round(pTrain*instanceCount1));
valInd = idx(round(pTrain*instanceCount1)+1:round((pTrain+pVal)*instanceCount1));
testInd = idx(round((pTrain+pVal)*instanceCount1)+1:end);

TrainingSet = inputData1(trainInd,:);
TestSet = inputData1(testInd,:);
ValidationSet = inputData1(valInd,:);

TrainingTargets = targets1(trainInd,:);
TestTargets = targets1(testInd,:);
ValidationTargets = targets1(valInd,:);

ValidationData = {ValidationSet,ValidationSet};

layers = [ sequenceInputLayer(featureCount, 'Name', 'in')
    lstmLayer(40, 'Name', 'lstm1')
    %eluLayer('Name', 'relu1')
    fullyConnectedLayer(featureCount, 'Name', 'lstm4')
    %reluLayer('Name', 'relu4')
    regressionLayer('Name', 'out') ];


% Set Training Options
options = trainingOptions('adam', ...
    ...'Plots', 'training-progress', ...
    'Shuffle','every-epoch',...
    'ValidationData', ValidationData, ...
    'ValidationFrequency', 20, ...
    'MiniBatchSize', 50,...
    'MaxEpochs',maxEpochs);

[net,info] = trainNetwork(TrainingSet, TrainingSet, layers, options);

%deepNetworkDesigner(net);
%assembledNet = assembleNetwork( layers_1 ) 

%% Test Error Calculation

preds = predict(net, TestSet);
% Calculate rmse
predsMat = cell2mat(preds')';
testMat = cell2mat(TestSet')';

m1 = mean((testMat-predsMat).^2,2);
rmse = mean(sqrt(m1));


%% Üst üste çizdirme
load('c8.mat')
wsize = 100;
Z = zscore(c8)';
ZT=Z';
predsWhole = predict(net, prepareDataWindows(Z, wsize, wsize));
predsMatWhole = cell2mat(predsWhole');
sample = [ZT(1:length(predsMatWhole),6),predsMatWhole(6,:)'];
figure
plot(sample)
legend(["actual" "constructed"])

sample2 = [ZT(1:length(predsMatWhole),7),predsMatWhole(7,:)'];
figure
plot(sample2)
legend(["actual" "constructed"])

%%