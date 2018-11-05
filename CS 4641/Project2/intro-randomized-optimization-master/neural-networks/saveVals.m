%Save dataset in correct format 
[num,txt,raw] = xlsread("pulsarOrNot.xlsx");

% [trainInd,valInd,testInd] = dividerand(Q,trainRatio,valRatio,testRatio)
[train_ind, val_ind, test_ind] = dividerand(17898, 0.7, 0.15, 0.15);
ans = 1;

% Result
t = num(:, 9);
temp = mod(t + 1, 2);
t = [t, temp];
t = t.';

% Rest of the data
x = num(:, 1:8);
x = x.';

save('pulsar_data.mat','ans','x', 't', 'train_ind', 'val_ind', 'test_ind');