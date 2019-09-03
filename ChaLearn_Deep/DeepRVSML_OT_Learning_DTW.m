function [traindownset,testdownsetdata] = DeepRVSML_OT_Learning_DTW(trainset,testsetdata,templatenum,options)

max_nIter = options.max_iters;
err_limit = options.err_limit;

classnum = length(trainset);
downdim = classnum*templatenum;
dim = size(trainset{1}{1},2);
middle_dim = 1024;
out_dim = templatenum*classnum;

trainsetnum = zeros(1,classnum);
virtual_sequence = cell(1,classnum);
active_dim = 0;
for c = 1:classnum
    trainsetnum(c) = length(trainset{c});
    virtual_sequence{c} = zeros(templatenum,downdim);
    for a_d = 1:templatenum
        active_dim = active_dim + 1;
        virtual_sequence{c}(a_d,active_dim) = 1;
    end
end

%% inilization
traindatafull = [];
labeldatafull = [];
for c = 1:classnum
    for n = 1:trainsetnum(c)
        seqlen = size(trainset{c}{n},1);
        T_ini = zeros(seqlen,templatenum);
        if seqlen>templatenum
            pernum = max(1,floor(seqlen/templatenum));
            index_start = 1;
            for temp_index = 1:templatenum-1
                index_end = index_start + pernum -1;
                T_ini(index_start:index_end,temp_index) = 1;
                index_start = index_end + 1;
            end
            T_ini(index_start:end,templatenum) = 1;
        else
            pernum = max(1,floor(templatenum/seqlen));
            index_start = 1;
            for temp_index = 1:seqlen-1
                index_end = index_start + pernum -1;
                T_ini(temp_index,index_start:index_end) = 1;
                index_start = index_end + 1;
            end
            T_ini(seqlen,index_start:end) = 1;
        end
        
        temp_label = [T_ini zeros(seqlen,1)+c-1];
        traindatafull = [traindatafull; trainset{c}{n}];
        labeldatafull = [labeldatafull; temp_label];
    end
end
save('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/traindatafull.mat','traindatafull','labeldatafull');
system('echo $LD_LIBRARY_PATH')

%% update
loss_old = 10^8;
for nIter = 1:max_nIter
    %matlab -nodesktop -nosplash -r 
    %system('source activate PL')
    if nIter==1
        system(['unset LD_LIBRARY_PATH; unset MKL_NUM_THREADS; unset MKL_DOMAIN_NUM_THREADS; python trainseq.py --data seqfull --data_root /data/Bing/Deep_metric-seq-master/datamat/ChaLearn --loss RVSML --in_dim ' num2str(dim) ' --middle_dim ' num2str(middle_dim) ' --out_dim ' num2str(out_dim) ' --L ' num2str(templatenum) ' --save_dir ./trainedmodels/ChaLearn --train_flag True'])
    else
        system(['unset LD_LIBRARY_PATH; unset MKL_NUM_THREADS; unset MKL_DOMAIN_NUM_THREADS; python trainseq.py --data seqfull --data_root /data/Bing/Deep_metric-seq-master/datamat/ChaLearn --loss RVSML --in_dim ' num2str(dim) ' --middle_dim ' num2str(middle_dim) ' --out_dim ' num2str(out_dim) ' --L ' num2str(templatenum) ' --save_dir ./trainedmodels/ChaLearn --resume ./trainedmodels/ChaLearn/ckp_ep500.pth.tar --train_flag True'])
    end
       
    %Pre_deepdata_Reverse
    load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/TransFeatures.mat');
    traindownset = Pre_deepdata_Reverse_aus(classnum, train_feature, trainsetnum, trainset);
    
    loss = 0;    
    %traindatafull = [];
    labeldatafull = [];
    N = sum(trainsetnum);
    for c = 1:classnum
        for n = 1:trainsetnum(c)
            seqlen = size(trainset{c}{n},1);
            %[dist, T] = OPW_w(traindownset{c}{n},virtual_sequence{c},[],[],lambda1,lambda2,delta,0);
            [dist, T] = dtw2((traindownset{c}{n})',virtual_sequence{c}');
            if isnan(dist)
                disp('Wrong Matching!')
            end
            loss = loss + dist;
            temp_label = [T zeros(seqlen,1)+c-1];
            labeldatafull = [labeldatafull; temp_label];
        end
    end
    
    save('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/traindatafull.mat','traindatafull','labeldatafull');
    
    loss = loss/N;
    if abs(loss - loss_old) < err_limit
        break;
    else
        loss_old = loss;
    end
 
end

testdatafull = [];
testsetdatanum = length(testsetdata);
for i = 1:testsetdatanum
    testdatafull = [testdatafull; testsetdata{i}];
end

save('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/testdatafull.mat','testdatafull');


%system('source activate PL')
system(['unset LD_LIBRARY_PATH; unset MKL_NUM_THREADS; unset MKL_DOMAIN_NUM_THREADS; python trainseq_final.py --data seqfull --data_root /data/Bing/Deep_metric-seq-master/datamat/ChaLearn --loss RVSML --in_dim ' num2str(dim) ' --middle_dim ' num2str(middle_dim) ' --out_dim ' num2str(out_dim) ' --L ' num2str(templatenum) ' --save_dir ./trainedmodels/ChaLearn --train_flag False'])
%system('source deactivate PL') unset MKL_NUM_THREADS; 

load('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/TransFeatures.mat');
%Pre_deepdata_Reverse_full
[traindownset,testdownsetdata] = Pre_deepdata_Reverse_full_aus(classnum, train_feature, test_feature, trainsetnum, trainset, testsetdatanum, testsetdata);
save('/data/Bing/Deep_metric-seq-master/datamat/ChaLearn/trainfull_pro_rvsml.mat','traindownset','testdownsetdata');
