
charnum = 20;
classnum = charnum;
dim = 60;
CVAL = 1;

% add path
addpath('E:/BING/ActionRecognition/FrameWideFeatures/vlfeat-0.9.18/toolbox');
vl_setup();
addpath('E:/BING/ActionRecognition/FrameWideFeatures/libsvm-3.20/matlab');

delta = 1;
lambda1 = 50;
lambda2 = 0.1;
options.max_iters = 200;
options.err_limit = 10^(-6);
options.lambda1 = lambda1;
options.lambda2 = lambda2;
options.delta = delta;

load MSR_Python_ori.mat;
trainset_m = trainset;
testsetdata_m = testsetdata;
testsetlabel = testsetdatalabel;

templatenum = 4;
lambda = 0.01;
tic;
L = RVSML_OT_Learning(trainset,templatenum,lambda,options);
RVSML_opw_time = toc;
%% classification with the learned metric
traindownset = cell(1,classnum);
testdownsetdata = cell(1,testsetdatanum);
for j = 1:classnum
    traindownset{j} = cell(trainsetnum(j),1);
    for m = 1:trainsetnum(j)
        traindownset{j}{m} = trainset{j}{m} * L;
    end
end
for j = 1:testsetdatanum
    testdownsetdata{j} = testsetdata{j} * L;
end
[RVSML_opw_map,RVSML_opw_acc,RVSML_opw_time] = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
RVSML_opw_acc_1 = RVSML_opw_acc(1);

templatenum = 4;
lambda = 0.1;
tic;
L = RVSML_OT_Learning_dtw(trainset,templatenum,lambda,options);
RVSML_dtw_time = toc;

%% classification with the learned metric
traindownset = cell(1,classnum);
testdownsetdata = cell(1,testsetdatanum);
for j = 1:classnum
    traindownset{j} = cell(trainsetnum(j),1);
    for m = 1:trainsetnum(j)
        traindownset{j}{m} = trainset{j}{m} * L;
    end
end
for j = 1:testsetdatanum
    testdownsetdata{j} = testsetdata{j} * L;
end
[RVSML_dtw_map,RVSML_dtw_acc,RVSML_dtw_time] = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
RVSML_dtw_acc_1 = RVSML_dtw_acc(1);

fprintf('Training time of RVSML instantiated by DTW is %.4f \n',RVSML_dtw_time);
fprintf('Classification using 1 nearest neighbor classifier with DTW distance:\n');
fprintf('MAP is %.4f \n',RVSML_dtw_map);
fprintf('Accuracy is %.4f \n',RVSML_dtw_acc_1);

fprintf('Training time of RVSML instantiated by OPW is %.4f \n',RVSML_opw_time);
fprintf('Classification using 1 nearest neighbor classifier with OPW distance:\n');
fprintf('MAP is %.4f \n',RVSML_opw_map);
fprintf('Accuracy is %.4f \n',RVSML_opw_acc_1);