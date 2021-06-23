function [] = prepareCSEN(Dic, param)    
    % Necessary variables for CSEN training and testing.
    maskM = param.maskM;
    maskN = param.maskN;
    trainLabel = param.trainLabel;
    testLabel = param.testLabel;
    dicRealLabel = param.dicRealLabel;
    trainRealLabel = param.trainRealLabel;
    testRealLabel = param.testRealLabel;
    
    Y0 = param.Y0;
    Y1 = param.Y1;
    Y2 = param.Y2;
    Proj_M = param.Proj_M;
    
    prox_Y0 = Proj_M * Y0;
    prox_Y1 = Proj_M * Y1;
    prox_Y2 = Proj_M * Y2;
    
    x_dic = zeros(length(Dic.label), maskM, maskN);
    x_train = zeros(length(trainLabel), maskM, maskN);
    x_test = zeros(length(testLabel), maskM, maskN);
    y_dic = zeros(length(Dic.label), maskM, maskN);
    y_train = zeros(length(trainLabel), maskM, maskN);
    y_test = zeros(length(testLabel), maskM, maskN);

    for i=1:length(Dic.label)
        x_dic(i,:,:) = reshape(prox_Y0(:, i), maskM, maskN);
        y_dic(i,:,:)=(Dic.label_matrix == Dic.label(i));
    end

    for i=1:length(trainLabel)
        x_train(i,:,:) = reshape(prox_Y1(:, i), maskM, maskN);
        y_train(i,:,:) = (Dic.label_matrix == trainLabel(i));
    end

    for i=1:length(testLabel)
        x_test(i,:,:) = reshape(prox_Y2(:, i), maskM, maskN);
        y_test(i,:,:)=(Dic.label_matrix==testLabel(i));
    end
    
    if ~exist(['CSENdata-', param.DicDesign], 'dir')
       mkdir(['CSENdata-', param.DicDesign])
    end    
    save(strcat(['CSENdata-', param.DicDesign], '/', param.modelName, '_mr_', num2str(param.MR), ...
        '_run', num2str(param.k), ('.mat')), ...
        'dicRealLabel', 'trainRealLabel', 'testRealLabel', ...
        'x_train', 'x_test', 'y_train', 'y_test', 'x_dic', 'y_dic', ...
        'Proj_M', 'Y0', 'Y1', 'Y2', '-v6');
    
    Dic.label_matrix;
    save(['CSENdata-', param.DicDesign, '/dic_label.mat'], 'ans');
   
end

