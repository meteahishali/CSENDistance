function [ Dic, train, test ] = split_data( data, labels, param, reallabel)
% Split the data for 5 different runs and build the dictionary.

class_n=length(unique(labels)); %number of class (people)
N1=param.dictionary_size; 
N2=param.train_size;
N3=param.test_size;

% Compute the size of group representation coefficients in 2-D plane:
% (m11, m22)
gds=[1; unique(cumprod(perms(factor(class_n)),2))];
m1=gds(end-2);
m2=class_n/m1;
gds=[1; unique(cumprod(perms(factor(N1)),2))];
m22=gds(floor((length(gds)-2)/2)+1);
m11=N1/m22;

% Initilization of variables.
D=zeros(size(data,1), N1 * class_n);
DicLabel = zeros(N1 * class_n, 1);

remSamples = length(labels) - N1 * class_n;
trainSize = round(N2 * (remSamples / (N2 + N3)));
testSize = remSamples - trainSize;
train.data = zeros(size(data, 1), trainSize);
test.data =  zeros(size(data, 1), testSize);
train.label=zeros(trainSize, 1);
train.reallabel=zeros(trainSize, 1); % For unquantized data.
test.label=zeros(testSize, 1);
test.reallabel=zeros(testSize, 1); % For unquantized data.

% Construct A matrix (label matrix).
meters = unique(labels);
temp= [];
A=[];
t=1;
for k=1:m2
    for l=1:m1
        temp=[temp,ones(m11,m22)*t];
        t=t+1;
    end
    A=[A;temp];
    temp=[];    
end

% Construct dictionary, train, and test data.
startTrain = 0;
startTest = 0;
for i=1:class_n
    % Sizes per class.
    NuSamples = sum(labels==meters(i));
    remSamplesClass = NuSamples - N1;
    trainSizeClass = round(N2 * (remSamplesClass / (N2 + N3)));
    testSizeClass = remSamplesClass - trainSizeClass;
    
    in=find(A(:)==i);
    in2=find(labels==meters(i));

    in2=in2(randperm(length(in2)));
    for k=1:length(in)
        D(:,in(k))=data(:,in2(k));
        DicLabel(in(k)) = reallabel(in2(k));
%        figure(99),imshow(reshape(D(:,k-1),32,32),[])
    end
    
    for l=1:trainSizeClass
        startTrain = startTrain + 1;
        train.data(:, startTrain) = data(:, in2(k+l));
        train.label(startTrain) = i;
        train.reallabel(startTrain) = reallabel(in2(k+l));
    end
    for t=1:testSizeClass
        startTest = startTest + 1;
        test.data(:, startTest) = data(:,in2(k+l+t));
        test.label(startTest) = i;
        test.reallabel(startTest) = reallabel(in2(k+l+t));
    end
end

% Remove empty places.
en = sum(train.data.*train.data);
tmp = find(en == 0);
train.data(:, tmp) = [];
train.label(tmp) = [];
train.reallabel(tmp) = [];

en = sum(test.data.*test.data);
tmp = find(en == 0);
test.data(:, tmp) = [];
test.label(tmp) = [];
test.reallabel(tmp) = [];

Dic.dictionary=D;
Dic.label=A(:);
Dic.label_matrix=A;
Dic.reallabel = DicLabel;
end

