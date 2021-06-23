%%
% Lei Zhang, Meng Yang, and Xiangchu Feng,
% "Sparse Representation or Collaborative Representation: Which Helps Face Recognition?", in ICCV 2011.

% The author of the modified implementation: Mete Ahishali,
% Tampere University, Tampere, Finland.

%%
clc, clear, close all
addpath(genpath('crc'));
% Change the file name accordingly.
param.modelName = 'DenseNet121';
%1D or 2D corresponding to the traditional and proposed dictionary designs.
param.DicDesign = '2D';
inputData = strcat('features/features_max_', param.modelName, '.mat');
load(inputData)
objectFeatures = double(objectFeatures);
angles = gtd(:, 1);
meters = gtd(:, 2);
outName = strcat('competing_splits/', param.modelName);
if ~exist('competing_splits/', 'dir')
   mkdir('competing_splits/')
end    
%% Pre-processing: Quantization and Sample Selection

% Samples between [0.5, 50.5] in meters. Quantization with 100 cms.
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

% Let compute minimum samples per class.
histt = zeros(length(unique(meters_quant)), 1);
for i = 1:length(unique(meters_quant))
   histt(i) = sum(meters_quant == i);
end
eqSize = min(histt);

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

for k = 1:nuR
    disp(['Run ' num2str(k) '.'])
    param.k = k;
    % For the competing methods combine dictionary samples with training
    % samples.
    xx_train = [Dic_all(k).dictionary'; train_all(k).data'];
    yy_train = [Dic_all(k).reallabel; train_all(k).reallabel];
    xx_test = test_all(k).data';
    yy_test = test_all(k).reallabel;

    % CRC
    [param.maskM, param.maskN] = size(Dic_all(k).label_matrix);
    N = size(Dic_all(k).dictionary, 1); % Size of the feature vector.

    Dic = Dic_all(k); % kth run.
    train = train_all(k);
    test = test_all(k);
     
    dicRealLabel = Dic.reallabel; % Unquantized labels.
    trainRealLabel = train.reallabel;
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

    % Measurements for dictionary.
    Y0 = phi * Dic.dictionary;
    energ_of_Y0 = sum(Y0.*Y0);
    tmp = find(energ_of_Y0 == 0);
    Y0(:,tmp)=[];
    train.label(tmp) = [];
    Y0 =  Y0./( repmat(sqrt(sum(Y0.*Y0)), [m,1]) ); %normalization

    % Measurments for training.
    Y1 = phi*train.data;
    energ_of_Y1=sum(Y1.*Y1);
    tmp=find(energ_of_Y1==0);
    Y1(:,tmp)=[];
    train.label(tmp)=[];
    Y1 = Y1./( repmat(sqrt(sum(Y1.*Y1)), [m,1]) ); %normalization

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
    ID = [];
    for indTest = 1:size(Y2,2)
        [id]    = CRC_RLS(A,Proj_M,Y2(:,indTest),Dic.label);
        ID      =   [ID id];
    end
    
    %%%%% Save variables
    param.Proj_M = Proj_M;
    param.Y0 = Y0;
    param.Y1 = Y1;
    param.Y2 = Y2;
    param.trainLabel = train.label;
    param.testLabel = test.label;
    param.dicRealLabel = dicRealLabel;
    param.trainRealLabel = trainRealLabel;
    param.testRealLabel = testRealLabel;
    
    % Compute necessary variables for CSEN training and testing.
    prepareCSEN(Dic, param);
    
    
    save(strcat(outName, '_mr_', num2str(param.MR), ...
        '_run', num2str(k), ('.mat')), ...
        'xx_train', 'xx_test', 'yy_train', 'yy_test', '-v6')
    
    ard(k) = sum(abs(ID' - test.reallabel)./test.reallabel) ...
                / length(test.reallabel);
    srd(k) = sum(((ID' - test.reallabel).^2)./test.reallabel) ...
                / length(test.reallabel);
    th(k, :) = max(test.reallabel./ ID', ID'./test.reallabel);
    rmse(k) = sqrt(sum((ID' - test.reallabel).^2) / length(test.reallabel));
    rmseLog(k) = sqrt(sum((log(ID') - log(test.reallabel)).^2) ...
                / length(test.reallabel));
    
    y_trues(k, :) = test.reallabel;
    y_preds(k, :) = ID';

end

outName_results = strcat('results/CRC_base/', param.modelName);
if ~exist('results/CRC_base', 'dir')
   mkdir('results/CRC_base/')
end
save([outName_results '_pred.mat'], 'y_trues', 'y_preds');

figure,
scatter(test.label, ID', 3, 'filled'), 
title(strcat('Collaborative Filtering, MSE: ', ...
    num2str(sum((test.label- ID').^2)/length(test.label))))
xlabel('Actual Distance in meters'), ylabel('Predicted Distance in meters')