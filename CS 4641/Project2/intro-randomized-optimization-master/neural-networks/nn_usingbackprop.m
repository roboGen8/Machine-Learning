%% LEARNING CURVE
clear all;
load pulsar_data.mat;

%Seed for randomization
setdemorandstream(100)

%Normalizing all the features to the range [0,1]
%[x,ps] = mapminmax(x, 0, 1); %ps contains the settings to normalize new data for consistency

%Initialize neural network
net = patternnet(5);

%NN training settings
net.divideFcn = 'divideind';

%Using a higher value of validation checks (300) to capture overfitting in plot
net.trainParam.max_fail = 300; 
net.trainParam.epochs = 1000;
net.divideParam.trainInd = train_ind;
net.divideParam.valInd = val_ind;
net.divideParam.testInd = test_ind;

%Training NN using Backprop Algo
[net, tr] = train(net, x, t);



%% COMPLEXITY CURVE

%Tuning for the right number of units in the hidden layer
best_val_err = ones(1,20);
nodes = zeros(1,20);
NET = cell(1,20);
TR = cell(1,20);
for i = 1:20
    nodes(1,i) = i*5;
    
    %Initializing NN
    net = patternnet(nodes(1,i));
    net.divideFcn = 'divideind';
    net.trainParam.max_fail = 300;
    net.trainParam.epochs = 1000;
    net.divideParam.trainInd = train_ind;
    net.divideParam.valInd = val_ind;
    net.divideParam.testInd = test_ind;   
    
    %Training NN
    [net, tr] = train(net, x, t);
    
    %Storing results
    best_val_err(1,i) = tr.best_vperf;
    NET{i} = net;
    TR{i} = tr;
end
plot(nodes(1,:), best_val_err(1,:));

%% BEST NETWORK PERFORMANCE
% Best found settings -----
% Single hidden layer; 15 Neurons; 1000 Epochs; 300 Validation checks;
% Training algo - Scaled Conjugate Gradient Backpropagation
best_net = NET{3};
best_tr = TR{3};
bep = best_tr.best_epoch;

trainX = x(:, train_ind);
trainT = t(:, train_ind);
trainY = round(best_net(trainX)); % Rounding converts probabilities into labels
testX = x(:, test_ind);
testT = t(:, test_ind);
testY = round(best_net(testX)); % Rounding converts probabilities into labels

sprintf('Total time taken - %0.2f %s', max(best_tr.time), 'seconds')
sprintf('Train error - %0.4f', crossentropy(best_net, trainT, best_net(trainX), {1}))
sprintf('Test error - %0.4f', crossentropy(best_net, testT, best_net(testX), {1}))
sprintf('Train classification acc. - %0.1f%%', 100*(1-(sum((trainY(:)-trainT(:)).^2)/(2*length(trainY)))))
sprintf('Test classification acc. - %0.1f%%', 100*(1-(sum((testY(:)-testT(:)).^2)/(2*length(testY)))))

