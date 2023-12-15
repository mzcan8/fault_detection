%bilstm classification

inputSize = 40;
numHiddenUnits = 40;
numClasses = 2;
maxEpochs = 100;
validationPatience = 3; % num. of validation

%outarr = [5 12];
outarr = [5 9 12 17];

[TrainingSet,inputData2,TrainingTargets,targets2] = prepareBilstmData(myDir, wsize, wshift, outarr);

instanceCount2 = length(inputData2);

pTrain2 = 0 ;
pVal2 = 0.50;
pTest2 = 0.50 ;
idx = randperm(instanceCount2);

valInd2 = idx(1:round(pVal2*instanceCount2));
testInd2 = idx(round(pVal2*instanceCount2)+1:end);

TestSet2 = inputData2(testInd2,:);
ValidationSet2 = inputData2(valInd2,:);

TestTargets2 = targets2(testInd2,:);
ValidationTargets2 = targets2(valInd2,:);

TrainingSet_40 = predict(assembledNet, TrainingSet);
TrainingTargetsCat = string(TrainingTargets);
TrainingTargetsCat = categorical(TrainingTargetsCat);

ValidationSet_40 = predict(assembledNet, ValidationSet2);
ValidationTargetsCat = string(ValidationTargets2);
ValidationTargetsCat = categorical(ValidationTargetsCat);

TestSet_40 = predict(assembledNet, TestSet2);
TestTargetsCat = string(TestTargets2);
TestTargetsCat = categorical(TestTargetsCat);

layers_for_classification = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options2 = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold', 1, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', 50, ...
    'ValidationData', {ValidationSet_40,ValidationTargetsCat}, ...
    'ValidationPatience', validationPatience, ...
    'SequenceLength', 'longest', ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'once', ...
    'Verbose', 0, ...
    'Plots', 'training-progress');


[net_for_classification, info_for_classification] = trainNetwork(TrainingSet_40,TrainingTargetsCat,layers_for_classification,options2);

figure
val_classes = classify(net_for_classification, ValidationSet_40);
cm = confusionchart(ValidationTargetsCat,val_classes);

figure
test_classes = classify(net_for_classification, TestSet_40);
cm2 = confusionchart(TestTargetsCat,test_classes);
acc = sum(test_classes == TestTargetsCat)./numel(TestTargetsCat);