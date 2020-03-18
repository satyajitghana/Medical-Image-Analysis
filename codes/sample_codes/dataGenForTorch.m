% Data preparation
src = dir('ALL_IDB2/img/*.tif'); % Path to folder containing the training images
data = zeros(length(src),3,32,32); % Initilaize data matrix (Nx3x32x32)
label = zeros(length(src),1); % Initialize label mtrix (Nx1)
for i = 1:length(src)
    filename = src(i).name;
    img = imread(filename); % Read image    
    img_32 = imresize(img,[32 32]); % Resize the image to 32x32 pixels
    % Transpose the image of dimensions MxNx3 to 3xMxN
    img_32 = permute(img_32,[3 1 2]);
    data(i,:,:,:) = reshape(img_32,[1 3 32 32]);
    % In the ALL_IDB2 dataset, class label is provoded in the image name
    % Extract the class label from the image name
    str1 = strsplit(filename,'_');
    str2 = strsplit(str1{1,2},'.');
    label(i,1) = str2num(str2{1,1})+1; % Converting string to number
end
% Shuffling data and label
n = randperm(size(data,1));
data = data(n,:,:,:);
label = label(n,1);
%% Use 80% of data for training, 10% for validation and 10% for testing
N_train = floor(0.8*size(data,1));
N_val = floor(0.9*size(data,1));
train_data = data(1:N_train,:,:,:); train_label = label(1:N_train,:);
val_data =  data(N_train+1:N_val,:,:,:); val_label = label(N_train+1:N_val,:);
test_data = data(N_val+1:end,:,:,:); test_label = label(N_val+1:end,:);
% Saving data and label files
save 'train_data' 'train_data'
save 'train_label' 'train_label'
save 'val_data' 'val_data'
save 'val_label' 'val_label'
save 'test_data' 'test_data'
save 'test_label' 'test_label'
