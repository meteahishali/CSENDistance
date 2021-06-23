%%
% Lei Zhang, Meng Yang, and Xiangchu Feng,
% "Sparse Representation or Collaborative Representation: Which Helps Face Recognition?", in ICCV 2011.

% The modified implementation is by Mete Ahishali and Mehmet Yamac,
% Tampere University, Tampere, Finland.

%%
clc, clear, close all
addpath(genpath('l1benchmark'));
addpath(genpath('l1magic-1'));
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
outName = ['results/', param.modelName];
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

%% Sparse Representation based Classification (CRC) implementation.
%%% Included SRC methods.
l1method={'solve_ADMM','solve_dalm','solve_OMP','solve_homotopy','solveGPSR_BCm', 'solve_L1LS','solve_l1magic','solve_PALM'}; %'solve_PALM' is very slow
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
    disp(['Run ' num2str(k) '.'])
    param.k = k; % kth run.
    
    N = size(Dic_all(k).dictionary, 1); % Size of the feature vector.

    Dic = Dic_all(k); 
    train = train_all(k);
    test = test_all(k);
    
    
    % Let compute minimum samples per class to build the dictionary.
    test.data = [train.data test.data];
    test.label = [train.label; test.label];
    test.reallabel = [train.reallabel; test.reallabel];
    
    histt = zeros(length(unique(test.label)), 1);
    for i = 1:length(unique(test.label))
        histt(i) = sum(test.label == i);
    end
    eqSize = min(histt);
    
    % We enlarge the dictionary by adding more samples.
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
    
    dicRealLabel = Dic.reallabel; % Unquantized labels with 4800 samples.
    testRealLabel = test.reallabel; % 33507 test samples.

    D = Dic.dictionary; %This is the dictionary with 4800 samples.
    
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
    
    test_length=length(test.label);
    
    for i=1:length(l1method)
        ID = [];
        tic
        for indTest = 1:test_length
            if ~mod(indTest,floor(test_length * 2/100))
                fprintf('\b\b\b\b\b\b%05.2f%%', indTest / test_length * 100);
            end
            [id]    =  L1_Classifier(A,Y2(:,indTest),Dic.label,l1method{i});
            ID      =   [ID id];
        end
        per.telapsed(k) = toc;
             
          
        per.ard(i, k) = sum(abs(ID' - test.label)./test.label) ...
                    / length(test.label);
        per.srd(i, k) = sum(((ID' - test.label).^2)./test.label) ...
                    / length(test.label);
        per.th(i, k, :) = max(test.label./ ID', ID'./test.label);
        per.rmse(i, k) = sqrt(sum((ID' - test.label).^2) / length(test.label));
        per.rmseLog(i, k) = sqrt(sum((log(ID') - log(test.label)).^2) ...
                        / length(test.label));

        per.y_trues(i, k, :) = test.label;
        per.testRealLabel(i, k, :) = test.reallabel;
        per.y_preds(i, k, :) = ID';
    end
end

 save([outName, '.mat'], 'per', '-v6')