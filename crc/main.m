%%
% Lei Zhang, Meng Yang, and Xiangchu Feng,
% "Sparse Representation or Collaborative Representation: Which Helps Face Recognition?", in ICCV 2011.

% The modified implementation is performed by Mete Ahishali,
% Tampere University, Tampere, Finland.

%%
clc, clear, close all
% Change the file name accordingly.
param.modelName = 'DenseNet121';
inputData = strcat('../features/features_max_', param.modelName, '.mat');
load(inputData)
objectFeatures = double(objectFeatures);
angles = gtd(:, 1);
meters = gtd(:, 2);
if ~exist('results/', 'dir')
    mkdir('results/')
end
CRC_light = 0; % 1 for CRC_light model or 0 for CRC.
if CRC_light == 1
    outName = ['results/light_', param.modelName];
else
    outName = ['results/', param.modelName];
end
%% Pre-processing: Quantization and Sample Selection

% Samples between [0.5, 60.5] in meters. Quantization with 100 cms.
partition = 0.5:1:60.5;

codebook = zeros(length(partition) + 1, 1);
codebook(2:length(partition) + 1) = 1:length(partition);
codebook(1) = -1;
codebook(end) = -1;
[~, meters_quant] = quantiz(meters, partition, codebook);

% Remove out of range samples
objectFeatures(meters_quant == -1, :) = [];
meters(meters_quant == -1, :) = [];
meters_quant(meters_quant == -1, :) = [];

%% Collaborative Representation based Classification (CRC) implementation.

data = objectFeatures';
label = meters_quant;
reallabel = meters;

param.dictionary_size = 20; % Samples per class in the dictionary.
param.train_size = 1; % These are the proportations. 1:1
param.test_size = 1;

nuR = 5; % Number of runs.
param.MR = 0.5; % Measurement rate.

measurement_type = 'eigen'; % Gauss, eigen, or None. None means no compression.
projection_matrix = 'l2_norm'; % minimum_norm or l2_norm.

rng(1)
[Dic_all(1), train_all(1), test_all(1)] = split_data(data,label,param,reallabel);
[Dic_all(2), train_all(2), test_all(2)] = split_data(data,label,param,reallabel);
[Dic_all(3), train_all(3), test_all(3)] = split_data(data,label,param,reallabel);
[Dic_all(4), train_all(4), test_all(4)] = split_data(data,label,param,reallabel);
[Dic_all(5), train_all(5), test_all(5)] = split_data(data,label,param,reallabel);

% Metrics.
ard = zeros(1, nuR);
srd = zeros(1, nuR);
th = zeros(nuR, length(test_all(1).label));
rmse = zeros(1, nuR);
rmseLog = zeros(1, nuR);
y_preds = zeros(nuR, length(test_all(1).label));
y_trues = zeros(nuR, length(test_all(1).label));

for k = 1:5
    disp(['Run ' num2str(k) '.'])
    disp(['Run ' num2str(k) '.'])
    param.k = k; % kth run.
    
    N = size(Dic_all(k).dictionary, 1); % Size of the feature vector.
    
    Dic = Dic_all(k);
    train = train_all(k);
    test = test_all(k);
    
    
    % Let compute minimum samples per class.
    test.data = [train.data test.data];
    test.label = [train.label; test.label];
    test.reallabel = [train.reallabel; test.reallabel];
    
    histt = zeros(length(unique(test.label)), 1);
    for i = 1:length(unique(test.label))
        histt(i) = sum(test.label == i);
    end
    eqSize = min(histt);
    
    if CRC_light == 0
        rng(k);
        for i = 1:max(test.label)
            indices = find(test.label == i);
            ind = randperm(length(indices));
            ind = ind(1:eqSize);
            Dic.dictionary = [Dic.dictionary test.data(:, indices(ind))];
            Dic.label = [Dic.label; test.label(indices(ind))];
            Dic.reallabel = [Dic.reallabel; test.reallabel(indices(ind))];
            
            test.data(:, indices(ind))= [];
            test.label(indices(ind))= [];
            test.reallabel(indices(ind))= [];
        end
    end
    
    dicRealLabel = Dic.reallabel; % Unquantized labels.
    testRealLabel = test.reallabel;
    
    D = Dic.dictionary; %This is the dictionary.
    
    m = floor(param.MR * N); % number of measurements
    
    % Dimensional reduction: measurement matrix Phi.
    switch measurement_type
        case 'eigen'
            [phi,disc_value,Mean_Image]  =  Eigen_f(D,m);
            phi = phi';
        case 'Gauss'
            phi = randn(m, N);
        case 'None'
            m = 1;
            phi = 1;
            param.MR = 1;
    end
    
    A  =  phi*D;
    A  =  A./( repmat(sqrt(sum(A.*A)), [m,1]) ); %normalization
    
    % Measurments for test.
    Y2 = phi*test.data;
    energ_of_Y2=sum(Y2.*Y2);
    tmp=find(energ_of_Y2==0);
    Y2(:,tmp)=[];
    test.label(tmp)=[];
    Y2 = Y2./( repmat(sqrt(sum(Y2.*Y2)), [m,1]) ); %normalization
    
    % Projection matrix computing
    kappa             =   0.4; % l2 regularized parameter value
    switch projection_matrix
        case 'minimum_norm'
            Proj_M=  pinv(A);
        case 'l2_norm'
            Proj_M = (A'*A+kappa*eye(size(A,2)))\A'; %l2 norm
        case 'transpose'
            Proj_M=  A';
    end
    
    %%%% Testing with CRC.
    test_length = size(Y2, 2);
    ID = zeros(1, test_length);
    tic
    for indTest = 1:test_length
        ID(1, indTest)    = CRC_RLS(A,Proj_M,Y2(:,indTest),Dic.label);
    end
    
    per.telapsed(k) = toc;
    
    per.ard(k) = sum(abs(ID' - test.label)./test.label) ...
        / length(test.label);
    per.srd(k) = sum(((ID' - test.label).^2)./test.label) ...
        / length(test.label);
    per.th(k, :) = max(test.label./ ID', ID'./test.label);
    per.rmse(k) = sqrt(sum((ID' - test.label).^2) / length(test.label));
    per.rmseLog(k) = sqrt(sum((log(ID') - log(test.label)).^2) ...
        / length(test.label));
    
    per.y_trues(k, :) = test.label;
    per.testRealLabel(k, :) = test.reallabel;
    per.y_preds(k, :) = ID';
end

save([outName, '.mat'], 'per', '-v6')

% figure,
% scatter(test.label, ID', 3, 'filled'),
% title(strcat('Collaborative Filtering, MSE: ', ...
%     num2str(sum((test.label- ID').^2)/length(test.label))))
% xlabel('Actual Distance in meters'), ylabel('Predicted Distance in meters')