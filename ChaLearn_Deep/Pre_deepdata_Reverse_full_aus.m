function [traindownset,testdownsetdata] = Pre_deepdata_Reverse_full_aus(classnum, train_feature, test_feature, trainsetnum, trainset, testsetdatanum, testsetdata)
traindownset = cell(1,classnum);
for c = 1:classnum
    traindownset{c} = cell(1,trainsetnum(c));
end

pos_s = 1;
for c = 1:classnum
    for i = 1:trainsetnum(c)
        pos_e = pos_s + size(trainset{c}{i},1) -1;
        traindownset{c}{i} = double(train_feature(pos_s:pos_e,:));
        pos_s = pos_e + 1;
    end
end
pos_s

testsetdatanum
testdownsetdata = cell(1,testsetdatanum);
pos_s = 1;
for i = 1:testsetdatanum
    pos_e = pos_s + size(testsetdata{i},1) -1;
    testdownsetdata{i} = double(test_feature(pos_s:pos_e,:));
    pos_s = pos_e + 1;
end
pos_s

end