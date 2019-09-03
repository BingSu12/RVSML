
charnum = 20;
classnum = charnum;
dim = 100;
%rankdim = 58;
CVAL = 1;


% TODO: add path
addpath('/home/sub/vlfeat-0.9.21/toolbox');
vl_setup();
addpath('/home/sub/libsvm-3.23/matlab');
%addpath('/home/sub/anaconda3/envs/PL/bin','/home/sub/bin','/home/sub/.local/bin','/home/sub/anaconda3/bin','/usr/local/sbin:/usr/local/bin','/usr/sbin','/usr/bin','/sbin','/bin','/usr/games','/usr/local/games','/snap/bin');
%addpath(genpath('/home/sub/anaconda3/envs/PL/Lib/site-packages/torch/lib'))  %D:\Anaconda3\envs\torch\Lib\site-packages\torch\lib

delta = 1;
lambda1 = 50;
lambda2 = 0.1;
options.max_iters = 50;
options.err_limit = 10^(-6);
options.lambda1 = lambda1;
options.lambda2 = lambda2;
options.delta = delta;

load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/trainset.mat');
load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/trainsetnum.mat');
load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/testset.mat');
load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/testsetnum.mat');

load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/testsetdata.mat');
load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/testsetlabel.mat');
load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/testsetdatanum.mat');

load(['/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/traindatamean.mat']);

trainset_m = trainset;
testsetdata_m = testsetdata;


%% RVSML-DTW
templatenum = 4;
[traindownset,testdownsetdata] = DeepRVSML_OT_Learning_DTW(trainset_m,testsetdata_m,templatenum,options);

[RVSML_dtw_NN_map,RVSML_dtw_NN_acc,RVSML_dtw_knn_time] = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
RVSML_dtw_NN_acc_1 = RVSML_dtw_NN_acc(1)
RVSML_dtw_NN_map


%% RVSML-OPW
templatenum = 4;
[traindownset,testdownsetdata] = DeepRVSML_OT_Learning_OPW(trainset_m,testsetdata_m,templatenum,options);

[RVSML_opw_NN_map,RVSML_opw_NN_acc,RVSML_opw_knn_time] = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
RVSML_opw_NN_acc_1 = RVSML_opw_NN_acc(1)
RVSML_opw_NN_map

fprintf('Deep-RVSML instantiated by DTW:');
fprintf('Classification using 1 nearest neighbor classifier with DTW distance:\n');
fprintf('MAP is %.4f \n',RVSML_dtw_NN_map);
fprintf('Accuracy is %.4f \n',RVSML_dtw_NN_acc_1);

fprintf('Deep-RVSML instantiated by OPW:');
fprintf('Classification using 1 nearest neighbor classifier with OPW distance:\n');
fprintf('MAP is %.4f \n',RVSML_opw_NN_map);
fprintf('Accuracy is %.4f \n',RVSML_opw_NN_acc_1);