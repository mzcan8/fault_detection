function X = prepareDataWindows(data, wsize, wshift)
%data: 5x1000 (5 features, 1000 instances)
[m,n] = size(data);
wcount = floor( (n - wsize) / wshift) + 1;
% Arrange data into cells
X = cell(wcount,1);
for i = 1:wcount
    wstart = (i-1)* wshift + 1;
    wend = wstart + wsize - 1;
    X{i,:} = data(:, wstart: wend);
end

end