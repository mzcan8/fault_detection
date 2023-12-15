function [inputData1,inputData2,outputData1,outputData2] = prepareBilstmData(datapath, wsize, wshift, outarr)

mins = 0.75;

%trs: time remaining size
trs = mins * 600;

inputData1 = cell(0,1);
outputData1 = cell(0,1);
inputData2 = cell(0,1);
outputData2 = cell(0,1);

myFiles = dir(fullfile(datapath,'*.mat'));

for j = 1:length(myFiles)
    baseFileName = myFiles(j).name;
    fullFileName = fullfile(datapath, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    fStruct = load(fullFileName);
    fCell = struct2cell(fStruct);
    flightData = fCell{1};
    Z=zscore(flightData);
    
    %Z: 1000x50 (50 features, 1000 instances)
    
    Z=Z';
    
    [m,n] = size(Z);
    df = 0;
    if n > trs %if j < 11 && n > trs
        df = n - trs;
        n = trs;
    end
    
    wcount = floor( (n - wsize) / wshift) + 1;
    % Arrange data into cells
    X = cell(wcount,1);
    Y = cell(wcount,1);
    
    lbl = 1;
    if j > 10
        lbl = 0;
    end
    
    for i = 1:wcount
        wstart = (i-1)* wshift + 1;
        wstart = wstart + df;
        wend = wstart + wsize - 1;
        X{i,:} = Z(:, wstart: wend);
        Y{i,:} = lbl;
    end
    %c4 = 5, h10 = 12
    if (ismember(j,outarr))
        inputData2 = [inputData2 ; X];
        outputData2 = [outputData2 ; Y];
    else
        inputData1 = [inputData1 ; X];
        outputData1 = [outputData1 ; Y];
    end
end

end