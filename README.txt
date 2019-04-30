                              Learning Distance for Sequences by Learning a Ground Metric

1. Introduction.

This package includes the prototype MATLAB codes and data for experiments on the MSR Action3D dataset, described in

	"Learning Distance for Sequences by Learning a Ground Metric", Bing Su and Ying Wu. ICML, 2019.

The 60-dimensional frame-wide features are employed as described in the supplementary file.

Tested under Windows 7 x64, Matlab R2015b.



2.  Usage & Dependency.

- MSRAction3D --- For the 60-dimensional frame-wide features processed by "https://github.com/InwoongLee/TS-LSTM" (described in "Lee, I., Kim, D., Kang, S., and Lee, S. Ensemble deep learning for skeleton-based action recognition using temporal sliding lstm networks. ICCV, 2017."), we use the proposed RVSML to learn the ground metric and employ the NN classifier to perform classification. We have provided an organized version of the features: https://pan.baidu.com/s/1pY25fGqmKhrSgprc7QynGw

  Dependency:
     vlfeat-0.9.18
     libsvm-3.20

  Usage:
     1. Download the file "MSR_Python_ori.mat" from "https://pan.baidu.com/s/1pY25fGqmKhrSgprc7QynGw" (Extraction code: inrn) and put it under this folder ("MSRAction3D"); 
     2. Add the installation paths of the two packages at the beginning of "EvaluateRVSML_MSRAction3D_60.m" below "TODO: add path"
     3. Run the main code in Matlab:
        EvaluateRVSML_MSRAction3D_60



3. License & disclaimer.

    The codes and data can be used for research purposes only. This package is strictly for non-commercial academic use only.



4. Notice

1) You may need to adjust the parameters when applying it on a new dataset. In some cases, you may need to normalize or centralize the input features in sequences. Please see the paper for details.

2) We utilized some toolboxes, data, and codes which are all publicly available. Please also check the license of them if you want to make use of this package.



5. Citations

Please cite the following papers if you use the codes:

Bing Su and Ying Wu, "Learning Distance for Sequences by Learning a Ground Metric," International Conference on Machine Learning, 2019.