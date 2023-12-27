function [inputData,outputData] = prepareLSTMData(datapath, wsize, wshift)

inputData = cell(0,1);
outputData = cell(0,1);

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
            wend = wstart + wsize - 1;
            X{i,:} = Z(:, wstart: wend);
            Y{i,:} = lbl;
        end
        
        inputData = [inputData ; X];
        outputData = [outputData ; Y];
        
end

end
