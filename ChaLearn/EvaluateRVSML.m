
charnum = 20;
classnum = charnum;
dim = 100;
%rankdim = 58;
CVAL = 1;


% TODO: add path
addpath('E:/BING/ActionRecognition/FrameWideFeatures/vlfeat-0.9.18/toolbox');
vl_setup();
addpath('E:/BING/ActionRecognition/FrameWideFeatures/libsvm-3.20/matlab');

delta = 1;
lambda1 = 50;
lambda2 = 0.1;
options.max_iters = 50;
options.err_limit = 10^(-2);
options.lambda1 = lambda1;
options.lambda2 = lambda2;
options.delta = delta;

load('./datamat/trainset.mat');
load('./datamat/trainsetnum.mat');
load('./datamat/testset.mat');
load('./datamat/testsetnum.mat');

load('./datamat/testsetdata.mat');
load('./datamat/testsetlabel.mat');
load('./datamat/testsetdatanum.mat');

load(['./datamat/traindatamean.mat']);
trainset_m = trainset;
for c=1:classnum
    for m = 1:trainsetnum(c)
        trainset_m{c}{m} = bsxfun(@minus, trainset{c}{m}, traindatamean);
    end
end
testsetdata_m = testsetdata;
for m = 1:testsetdatanum
    testsetdata_m{m} = bsxfun(@minus, testsetdata{m}, traindatamean);
end

%% RVSML-DTW
templatenum = 4;
lambda = 0.0005;
tic;
L = RVSML_OT_Learning_dtw(trainset_m,templatenum,lambda,options);
dtwtrain_time = toc

traindownset = cell(1,classnum);
testdownsetdata = cell(1,testsetdatanum);
for j = 1:classnum
    traindownset{j} = cell(trainsetnum(j),1);
    for m = 1:trainsetnum(j)
        traindownset{j}{m} = trainset_m{j}{m} * L;
    end
end
for j = 1:testsetdatanum
    testdownsetdata{j} = testsetdata_m{j} * L;
end
[RVSML_dtw_NN_map,RVSML_dtw_NN_acc,RVSML_dtw_knn_time] = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
RVSML_dtw_NN_acc_1 = RVSML_dtw_NN_acc(1);

%% RVSML-OPW
templatenum = 4;
lambda = 0.00005;
tic
L = RVSML_OT_Learning(trainset_m,templatenum,lambda,options);
opwtrain_time = toc
    
traindownset = cell(1,classnum);
testdownsetdata = cell(1,testsetdatanum);
for j = 1:classnum
    traindownset{j} = cell(trainsetnum(j),1);
    for m = 1:trainsetnum(j)
        traindownset{j}{m} = trainset_m{j}{m} * L;
    end
end
for j = 1:testsetdatanum
    testdownsetdata{j} = testsetdata_m{j} * L;
end
[RVSML_opw_NN_map,RVSML_opw_NN_acc,RVSML_opw_knn_time] = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
RVSML_opw_NN_acc_1 = RVSML_opw_NN_acc(1);

fprintf('Training time of RVSML instantiated by DTW is %.4f \n',dtwtrain_time);
fprintf('Classification using 1 nearest neighbor classifier with DTW distance:\n');
fprintf('MAP is %.4f \n',RVSML_dtw_NN_map);
fprintf('Accuracy is %.4f \n',RVSML_dtw_NN_acc_1);

fprintf('Training time of RVSML instantiated by OPW is %.4f \n',opwtrain_time);
fprintf('Classification using 1 nearest neighbor classifier with OPW distance:\n');
fprintf('MAP is %.4f \n',RVSML_opw_NN_map);
fprintf('Accuracy is %.4f \n',RVSML_opw_NN_acc_1);