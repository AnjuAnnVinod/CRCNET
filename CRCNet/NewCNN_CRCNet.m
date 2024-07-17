close all
clear
clc

%% READ TRAINING IMAGES
allImages = imageDatastore('./images1', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[training_set, validation_set, testing_set] = splitEachLabel(allImages, 0.4, 0.2, 0.4);

%% PREPARE AND MODIFY NEURAL NET
numClasses = numel(categories(training_set.Labels));
%% Create a simple custom CNN architecture
lgraph = layerGraph();
tempLayers = [
    imageInputLayer([224 224 3],"Name","data")
    convolution2dLayer([7 7],64,"Name","conv1-7x7_s2","BiasLearnRateFactor",2,"Padding",[3 3 3 3],"Stride",[2 2])
    reluLayer("Name","conv1-relu_7x7")
    maxPooling2dLayer([3 3],"Name","pool1-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])
    crossChannelNormalizationLayer(3,"Name","pool1-norm1","K",1)
    convolution2dLayer([1 1],64,"Name","conv2-3x3_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","conv2-relu_3x3_reduce")
    convolution2dLayer([3 3],192,"Name","conv2-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","BN1")
    leakyReluLayer(0.9,"Name","leakyRelu_1")
    crossChannelNormalizationLayer(3,"Name","conv2-norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution2dLayer([1 1],64,"Name","Conva-1x1","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BN2")
    reluLayer("Name","Conv-relu_3x3_a")
    convolution2dLayer([3 3],128,"Name","Conva-3x3F","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","BN23")
    leakyReluLayer(0.9,"Name","leakyRelu_2")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution2dLayer([1 1],96,"Name","Conv-3x3_reduce","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BNNN77")
    reluLayer("Name","Conv-relu_3x3_reduce")
    convolution2dLayer([3 3],256,"Name","Conva-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","BN3")
    leakyReluLayer(0.9,"Name","leakyRelu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","Conv-5x5_reduce","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BNNN66")
    reluLayer("Name","Conv-relu_5x5_reduce")
    convolution2dLayer([5 5],32,"Name","Conva-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    batchNormalizationLayer("Name","BN31")
    leakyReluLayer(0.9,"Name","leakyRelu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Conva-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","Conva-pool_proj","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BN4")
    leakyReluLayer(0.9,"Name","leakyRelu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","ConvConcat-output");
lgraph = addLayers(lgraph,tempLayers);
%% 2nd Set
tempLayers = [
    convolution2dLayer([1 1],128,"Name","Convb-1x1","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BNNNN66")
    reluLayer("Name","Convb-relu_3x3_reduceP")
    convolution2dLayer([3 3],192,"Name","Convb-3x3P","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","BN5")
    leakyReluLayer(0.9,"Name","leakyRelu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","Convb-3x3_reduce","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BNNN55")
    reluLayer("Name","Convb-relu_3x3_reduce")
    convolution2dLayer([3 3],192,"Name","Convb-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","BN6")
    leakyReluLayer(0.9,"Name","leakyRelu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","Convb-5x5_reduce","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BNNN44")
    reluLayer("Name","Convb-relu_5x5_reduce")
    convolution2dLayer([5 5],96,"Name","Convb-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    batchNormalizationLayer("Name","BN7")
    leakyReluLayer(0.9,"Name","leakyRelu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Convb-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","Convb-pool_proj","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BN8")
    leakyReluLayer(0.9,"Name","leakyRelu_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","ConvbConcat-output")
    maxPooling2dLayer([3 3],"Name","pool3-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

%%
%% third set
tempLayers = [
    convolution2dLayer([1 1],192,"Name","Conv_4a-1x1","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BNNN33")
    reluLayer("Name","Convb-relu_3x3_reduceT")
    convolution2dLayer([3 3],192,"Name","Convb-3x3T","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","BN9")
    leakyReluLayer(0.9,"Name","leakyRelu_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","Conv_4a-3x3_reduce","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BNNN22")
    reluLayer("Name","Conv_4a-relu_3x3_reduce")
    convolution2dLayer([3 3],208,"Name","Conv_4a-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","BN10")
    leakyReluLayer(0.9,"Name","leakyRelu_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","Conv_4a-5x5_reduce","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BNNN11")
    reluLayer("Name","Conv_4a-relu_5x5_reduce")
    convolution2dLayer([5 5],48,"Name","Conv_4a-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    batchNormalizationLayer("Name","BN11")
    leakyReluLayer(0.9,"Name","leakyRelu_11p5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","Conv_4a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","Conv_4a-pool_proj","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","BN12")
    leakyReluLayer(0.9,"Name","leakyRelu_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","Conv_4a-output");
lgraph = addLayers(lgraph,tempLayers);
%%
tempLayers = [
    convolution2dLayer([3 3], 192, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'BNNW1')
    leakyReluLayer(0.9,"Name", 'relu1')
    globalAveragePooling2dLayer('Name', 'Global_Average_Pooling1')
    
    convolution2dLayer([1 1], 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'BNNW2')
    leakyReluLayer(0.9,"Name", 'relu2')
    globalAveragePooling2dLayer('Name', 'Global_Average_Pooling2')
    
    dropoutLayer(0.7,"Name","ModifiedDropoutLayer")
    fullyConnectedLayer(9,"Name","FC_Layer","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
    softmaxLayer("Name","prob")
    classificationLayer("Name","OutputLayer")
];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"pool2-3x3_s2","Conva-1x1");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","Conv-3x3_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","Conv-5x5_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","Conva-pool");
lgraph = connectLayers(lgraph,"leakyRelu_2","ConvConcat-output/in1");
lgraph = connectLayers(lgraph,"leakyRelu_3","ConvConcat-output/in2");
lgraph = connectLayers(lgraph,"leakyRelu_4","ConvConcat-output/in3");
lgraph = connectLayers(lgraph,"leakyRelu_5","ConvConcat-output/in4");
lgraph = connectLayers(lgraph,"ConvConcat-output","Convb-1x1");
lgraph = connectLayers(lgraph,"ConvConcat-output","Convb-3x3_reduce");
lgraph = connectLayers(lgraph,"ConvConcat-output","Convb-5x5_reduce");
lgraph = connectLayers(lgraph,"ConvConcat-output","Convb-pool");
lgraph = connectLayers(lgraph,"leakyRelu_6","ConvbConcat-output/in1");
lgraph = connectLayers(lgraph,"leakyRelu_7","ConvbConcat-output/in2");
lgraph = connectLayers(lgraph,"leakyRelu_8","ConvbConcat-output/in3");
lgraph = connectLayers(lgraph,"leakyRelu_9","ConvbConcat-output/in4");
%lgraph = connectLayers(lgraph,"ConvbConcat-output","Conv_4a-pool");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","Conv_4a-1x1");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","Conv_4a-3x3_reduce");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","Conv_4a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","Conv_4a-pool");
lgraph = connectLayers(lgraph,"leakyRelu_10","Conv_4a-output/in1");
lgraph = connectLayers(lgraph,"leakyRelu_11","Conv_4a-output/in2");
lgraph = connectLayers(lgraph,"leakyRelu_11p5","Conv_4a-output/in3");
lgraph = connectLayers(lgraph,"leakyRelu_12","Conv_4a-output/in4");
lgraph = connectLayers(lgraph,"Conv_4a-output","conv1");
%% Introduce class weights
classes = ["ADI" "BACK" "DEB" "LYM" "MUC" "MUS" "NORM" "STR" "TUM"];
imageInputSize = lgraph.Layers(1, 1).InputSize;
clsLayer = classificationLayer('Name', 'OutputLayer', 'Classes',classes);
lgraphNew = replaceLayer(lgraph, "OutputLayer", clsLayer);

%% display the network
analyzeNetwork(lgraphNew)
plot(lgraphNew)
%% DATA AUGMENTATION FOR TRAINING
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter('RandRotation', [-20, 20], 'RandXReflection', true, 'RandYReflection', true, ...
    'RandXTranslation', pixelRange, 'RandYTranslation', pixelRange);
augmented_training_set = augmentedImageDatastore(imageInputSize, training_set, 'DataAugmentation', imageAugmenter);
resized_validation_set = augmentedImageDatastore(imageInputSize, validation_set, 'DataAugmentation', imageAugmenter);
resized_testing_set = augmentedImageDatastore(imageInputSize, testing_set, 'DataAugmentation', imageAugmenter);

%% TRAIN
miniBatchSize = 32;
valFrequency = floor(numel(augmented_training_set.Files) / (miniBatchSize));

% Adjust learning rate and other hyperparameters for 'adam' solver
opts = trainingOptions('rmsprop', ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 1e-4, ... % Adjust the learning rate
    'L2Regularization', 1e-5, ...
    'MaxEpochs', 20, ... % Increase the number of epochs
    'GradientThresholdMethod', 'l2norm', ...
    'Verbose', true, ...
    'ValidationFrequency', valFrequency, ...
    'ExecutionEnvironment', 'auto', ...
    'ValidationData', resized_validation_set, ...
    'Plots', 'training-progress');

net = trainNetwork(augmented_training_set, lgraphNew, opts);

%% TESTING Process
[predLabels, predScores] = classify(net, resized_testing_set, 'ExecutionEnvironment', 'auto');
figure, plotconfusion(testing_set.Labels, predLabels)
conf = confusionmat(testing_set.Labels, predLabels);
figure(), imagesc(conf);
axis square
colorbar
set(gcf, 'Color', 'w');
PerItemAccuracy = mean(predLabels == testing_set.Labels);
title(['overall per image accuracy ', num2str(round(100*PerItemAccuracy)), '%'])

%% -----------Code for ROC -----------------
cgt = double(testing_set.Labels); 
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(cgt,predScores(:,1),1);
figure,plot(X,Y,'LineWidth',3);
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification ')


