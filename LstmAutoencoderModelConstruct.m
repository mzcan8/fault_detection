myDir = uigetdir;
featureCount = 50;
maxEpochs = 200;
wsize = 10;
wshift = 5;

[inputData , targets] = prepareLSTMData(myDir, wsize, wshift);

instanceCount = length(inputData);

pTrain = 0.80 ;
pVal = 0.10;
pTest = 0.10 ;
idx = randperm(instanceCount);
trainInd = idx(1:round(pTrain*instanceCount));
valInd = idx(round(pTrain*instanceCount)+1:round((pTrain+pVal)*instanceCount));
testInd = idx(round((pTrain+pVal)*instanceCount)+1:end);

TrainingSet = inputData(trainInd,:);
TestSet = inputData(testInd,:);
ValidationSet = inputData(valInd,:);

ValidationData = {ValidationSet,ValidationSet};

layers = [ sequenceInputLayer(featureCount, 'Name', 'in')
    lstmLayer(40, 'Name', 'lstm3')
    %eluLayer('Name', 'relu1')
    fullyConnectedLayer(featureCount, 'Name', 'lstm4')
    %reluLayer('Name', 'relu4')
    regressionLayer('Name', 'out') ];


% Set Training Options
options = trainingOptions('adam', ...
    'Plots', 'training-progress', ...
    'Shuffle','every-epoch',...
    'ValidationData', ValidationData, ...
    'ValidationFrequency', 20, ...
    'MiniBatchSize', 50,...
    'MaxEpochs',maxEpochs);

[net,info] = trainNetwork(TrainingSet, TrainingSet, layers, options);

%To take encoder part of the network
%deepNetworkDesigner(net);
%assembledNet = assembleNetwork( layers_1 ) 

%% Test Error Calculation

preds = predict(net, TestSet);
% Calculate rmse
predsMat = cell2mat(preds')';
testMat = cell2mat(TestSet')';

m1 = mean((testMat-predsMat).^2,2);
rmse = mean(sqrt(m1));


%% Plot actual and reconstructed signals

%load('c11.mat')
wsize = 10;
Z = zscore(c10)';
ZT=Z';
predsWhole = predict(net, prepareDataWindows(Z, wsize, wsize));
predsMatWhole = cell2mat(predsWhole');
sample = [ZT(1:length(predsMatWhole),23),predsMatWhole(23,:)'];
figure
plot(sample)
legend(["actual" "reconstructed"])

%sample2 = [ZT(1:length(predsMatWhole),5),predsMatWhole(5,:)'];
%figure
%plot(sample2)
%legend(["actual" "constructed"])

 sz = length(predsMatWhole);
 m1 = mean((ZT(1:sz, :)-predsMatWhole').^2,1);
 rmseTemp = mean(sqrt(m1))

%%
