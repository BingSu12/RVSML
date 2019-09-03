function [Map,Acc,knn_averagetime] = NNClassifier(classnum,trainset,trainsetnum,testsetdata,testsetdatanum,testsetlabel,options)
    lambda1 = options.lambda1;
    lambda2 = options.lambda2;
    delta = options.delta;
    
    trainsetdatanum = sum(trainsetnum);
    trainsetdata = cell(1,trainsetdatanum);
    trainsetlabel = zeros(trainsetdatanum,1);
    sample_count = 0;
    for c = 1:classnum
        for per_sample_count = 1:trainsetnum(c)
            sample_count = sample_count + 1;
            trainsetdata{sample_count} = trainset{c}{per_sample_count};
            trainsetlabel(sample_count) = c;
        end
    end
    
    testsetlabelori = testsetlabel;
    testsetlabel = getLabel(testsetlabelori);
    trainsetlabelfull = getLabel(trainsetlabel);
 
    %save('./datamat/trainsetdata.mat','trainsetdata');
    %save('./datamat/trainsetlabel.mat','trainsetlabel');
    
    k_pool = [1 3 5 7 9 11 15 30];
    k_num = size(k_pool,2);
    Acc = zeros(k_num,1);
    %dtw_knn_map = zeros(vidsetnum,);
    %testsetlabelori = testsetlabel;
    %testsetlabel = getLabel(testsetlabelori);

    %trainsetdatanum = sum(trainsetnum);
    tic   
    dis_totrain_scores = zeros(trainsetdatanum,testsetdatanum);
    ClassLabel = [1:classnum]';
    dis_ap = zeros(1,testsetdatanum);
    
    rightnum = zeros(k_num,1);
    for j = 1:testsetdatanum       
        for m2 = 1:trainsetdatanum
            %[Dist,D,matchlength,w] = dtw2(trainsetdata{m2}',testsetdata{j}');
            %[Dist,T] = Sinkhorn_distance(trainsetdata{m2},testsetdata{j},lambda,0);     
            [Dist,T] = OPW_w(trainsetdata{m2},testsetdata{j},[],[],lambda1,lambda2,delta,0);
            %[dist, T] = OPW_w(ct_barycenter{c}.supp',ct_barycenter{c2}.supp',ct_barycenter{c}.w',ct_barycenter{c2}.w',lambda1,lambda2,delta,0);
            dis_totrain_scores(m2,j) = Dist;
            if isnan(Dist)
                disp('Not a number!');
            end
        end
        
        [distm,index]=sort(dis_totrain_scores(:,j));
        
        for k_count = 1:k_num
            cnt=zeros(classnum,1);
            for temp_i = 1:k_pool(k_count)
                ind=find(ClassLabel==trainsetlabel(index(temp_i)));
                cnt(ind)=cnt(ind)+1;
            end
            [distm2,ind]=max(cnt);        
            predict=ClassLabel(ind);
            predict = predict(1);
            if predict==testsetlabelori(j)
                rightnum(k_count) = rightnum(k_count) + 1;
            end
        end              
        
        temp_dis = -dis_totrain_scores(:,j);
        %temp_dis
        temp_dis(find(isnan(temp_dis))) = 0;
        [~,~,info] = vl_pr(trainsetlabelfull(:,testsetlabelori(j)),temp_dis);
        dis_ap(j) = info.ap;
    end
    Acc = rightnum./testsetdatanum;    
    Map = mean(dis_ap);
    
    knn_time = toc;
    knn_averagetime = knn_time/testsetdatanum;
end

function [X] = getLabel(classid)
    X = zeros(numel(classid),max(classid))-1;
    for i = 1 : max(classid)
        indx = find(classid == i);
        X(indx,i) = 1;
    end
end