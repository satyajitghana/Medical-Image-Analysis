%% Generating training and testing data
dataDirPath = 'ALL_IDB2/img/';
src = dir([dataDirPath,'*.tif']);
N = length(src);
feat = zeros(1028,N);
label = zeros(2,N);
for i = 1:N     
    filename = src(i).name;
    img = imread([dataDirPath,filename]); % Reading an image
    gray_img = rgb2gray(img); % Converting to grayscale
    glcm = graycoprops(gray_img); % Extarcting co-occurance matrix features
    % Re-rranging the features into a 1x4 vector
    feat1 = [glcm.Contrast;glcm.Correlation;glcm.Energy;glcm.Homogeneity]; %texture features

    % Preparing data for training NN
    img_32 = imresize(gray_img,[32 32]); % Resizing the image to 32x32 pixels
    img_32 = mat2gray(img_32); % Normalizing the image

    feat(:,i) = [img_32(:);feat1]; % 1028x1
    % Assigning class label assuming class 0 = healthy, class 1 = diseases
    str1 = strsplit(filename,'_');
    str2 = strsplit(str1{1,2},'.');
    class = str2{1,1};
    if class == '0'
        label(:,i) = [1 0];
    else
        label(:,i) = [0 1];
    end    

end

% Shuffling data 
n = randperm(N);
feat = feat(:,n);
label = label(:,n);
% Saving the features and labels for future use
%save([dataDirPath,'feat.mat'], 'feat');
%save([dataDirPath,'label.mat'], 'label');

%% Training network
% Load data
%load feat
%load label
% Initialize neural network for classification
net = patternnet(100); % NN with single hidden layer having 100 units
% Default setting uses 70% data for taining,15% for validation, 15% for testing
net = train(net,feat,label);

