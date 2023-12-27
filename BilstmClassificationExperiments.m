%bilstm classification

myDir = uigetdir;
aveValAccArr = zeros(1,10);
aveValFscoreArr = zeros(1,10);

for repeatcount = 1:10

inputSize = 30;%45;
numHiddenUnits = 30;
numClasses = 2;
maxEpochs = 100;
validationPatience = 7; % num. of validation

wsize = 10; %100
wshift = 5; %20

%k-fold
%k = 1;
k = 10;
fcount = 20;

arr1 = [4 10 9 2 3 8 6 7 1 5];%randperm(fcount/2);
arr2 = [20 12 18 17 19 14 15 11 16 13];%randperm(fcount/2)+10;
outcount = (fcount / k)/2;

valLossArr = zeros(1,k);
valAccArr = zeros(1,k);
accArr = zeros(1,k);

valPrecArr = zeros(1,k);
valRecArr = zeros(1,k);
valFscoreArr = zeros(1,k);

for i=1:k
    
    %outarr: one halthy one fulty selection
    st = (i-1)*outcount+1;
    outarr = [arr1(st:st+outcount-1) arr2(st:st+outcount-1)];
    
    %outarr = [7 8 14 17];
    
    [TrainingSet,inputData2,TrainingTargets,targets2] = prepareBilstmData(myDir, wsize, wshift, outarr);
    
    %instanceCount2 = length(inputData2);
    
    % pTrain2 = 0 ;
    % pVal2 = 0.50;
    % pTest2 = 0.50 ;
    % idx = randperm(instanceCount2);
    % 
    % valInd2 = idx(1:round(pVal2*instanceCount2));
    % testInd2 = idx(round(pVal2*instanceCount2)+1:end);
    
    TestSet2 = inputData2; %inputData2(testInd2,:);
    ValidationSet2 = inputData2 ;%inputData2(valInd2,:);
    
    TestTargets2 = targets2; %targets2(testInd2,:);
    ValidationTargets2 = targets2; %targets2(valInd2,:);
    %
    % if ValidationTargets2{1} == 1
    %     ValidationTargets2{1} = 0;
    % else
    %     ValidationTargets2{1} = 1;
    % end
    %
    TrainingSet_40 = predict(assembledNet, TrainingSet); % TrainingSet;
    TrainingTargetsCat = string(TrainingTargets);
    TrainingTargetsCat = categorical(TrainingTargetsCat);
    
    ValidationSet_40 = predict(assembledNet, ValidationSet2); % ValidationSet2; 
    ValidationTargetsCat = string(ValidationTargets2);
    ValidationTargetsCat = categorical(ValidationTargetsCat);
    
    TestSet_40 = predict(assembledNet, TestSet2); %TestSet2;
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
        'OutputNetwork', 'best-validation-loss', ...
        'SequenceLength', 'longest', ...
        'ValidationFrequency', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0); %, ...
        %'Plots', 'training-progress');
    
    
    [net_for_classification, info_for_classification] = trainNetwork(TrainingSet_40,TrainingTargetsCat,layers_for_classification,options2);
    
    %figure
    val_classes = classify(net_for_classification, ValidationSet_40);
    %cm = confusionchart(ValidationTargetsCat,val_classes);
    
    %figure
    test_classes = classify(net_for_classification, TestSet_40);
    %cm2 = confusionchart(TestTargetsCat,test_classes);
    acc = sum(test_classes == TestTargetsCat)./numel(TestTargetsCat);
    
    valLossArr(i) = info_for_classification.FinalValidationLoss;
    valAccArr(i) = info_for_classification.FinalValidationAccuracy;
    accArr(i) = acc;

    
    
end

aveValAcc = sum(valAccArr) / k;
aveAcc = sum(accArr) / k;

aveValAccArr(repeatcount) = aveValAcc;

end

final_Acc = mean(aveValAccArr)
