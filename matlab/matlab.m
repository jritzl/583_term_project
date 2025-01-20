% Load MNIST Dataset (Use a custom loader for CIFAR-10)
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Resize Images to a Vector
inputSize = [28 28]; % MNIST image size
imds.ReadFcn = @(x) imresize(imread(x), inputSize);
[trainData, testData] = splitEachLabel(imds, 0.8);

% Prepare Training Data
trainImages = zeros(numel(trainData.Files), prod(inputSize));
for i = 1:numel(trainData.Files)
    img = readimage(trainData, i);
    trainImages(i, :) = double(img(:))' / 255; % Vectorize and normalize
end
trainLabels = trainData.Labels;

% Train the SVM
svmModel = fitcecoc(trainImages, trainLabels, 'Coding', 'onevsall', 'Learners', 'Linear');

% Prepare Testing Data
testImages = zeros(numel(testData.Files), prod(inputSize));
for i = 1:numel(testData.Files)
    img = readimage(testData, i);
    testImages(i, :) = double(img(:))' / 255;
end
testLabels = testData.Labels;

% Predict and Evaluate
predictedLabels = predict(svmModel, testImages);
accuracy = mean(predictedLabels == testLabels);

disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

% Generate Confusion Matrix
confMat = confusionmat(testLabels, predictedLabels); % Numerical matrix
disp('Confusion Matrix:');
disp(confMat);

% Plot Confusion Matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for MNIST Classification with SVM');

% Specify the directory containing CIFAR-10 batches
dataDir = 'C:\Users\Yunus\Downloads\cifar-10-matlab\cifar-10-batches-mat';



% Load all training batches
trainImages = [];
trainLabels = [];
for batch = 1:5
    fileName = fullfile(dataDir, sprintf('data_batch_%d.mat', batch));
    [batchData, batchLabels] = loadBatch(fileName);
    trainImages = cat(4, trainImages, batchData);
    trainLabels = [trainLabels; batchLabels];
end

% Load test batch
[testImages, testLabels] = loadBatch(fullfile(dataDir, 'test_batch.mat'));

% Normalize pixel values to [0, 1]
trainImages = double(trainImages) / 255;
testImages = double(testImages) / 255;

disp('Data successfully loaded and preprocessed!');
% Function to load a single batch

% Flatten images for SVM
trainDataFlattened = reshape(trainImages, [], size(trainImages, 4))'; % [N x 3072]
testDataFlattened = reshape(testImages, [], size(testImages, 4))'; % [N x 3072]

% Train an SVM
svmModel = fitcecoc(trainDataFlattened, trainLabels, 'Coding', 'onevsall', 'Learners', 'Linear');

% Predict on the test set
predictedLabels = predict(svmModel, testDataFlattened);

% Generate confusion matrix
confMat = confusionmat(testLabels, predictedLabels);

% Display confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Visualize confusion matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for CIFAR-10 Classification');
% Predict and Evaluate


accuracy = mean(predictedLabels == testLabels);

disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

function [data, labels] = loadBatch(filePath)
    batch = load(filePath);
    data = reshape(batch.data, [], 32, 32, 3); % Reshape to 32x32x3
    data = permute(data, [2, 3, 4, 1]); % Rearrange dimensions to [32x32x3xN]
    labels = categorical(batch.labels); % Convert labels to categorical
end
