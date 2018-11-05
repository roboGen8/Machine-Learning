% INITIALIZE THE NEURAL NETWORK PROBLEM %
clear all;

load pulsar_data.mat;


% Number of neurons
n = 15; % recovered from the backprop step

% Number of attributes and number of classifications
[n_attr, ~]  = size(x);
[n_class, ~] = size(t);

% Initialize neural network
net = patternnet(n);

% Configure the neural network for this dataset
net = configure(net, x, t); 

%% SPLITTING DATASETS
trainX = x(:, train_ind);
trainT = t(:, train_ind);
valX = x(:, val_ind);
valT = t(:, val_ind);
testX = x(:, test_ind);
testT = t(:, test_ind);

%% 
%%%%% Defining the loss function
% w is the variable in the function defined below, which are the NN
% weights. 
fun = @(w) ce_test(w, net, trainX, trainT);

%%%%% Constructing the initial weight vector -
% There is n_attr attributes in dataset, and there are n neurons so there 
% are total of n_attr*n input weights (uniform weight)
initial_il_weights = ones(1, n_attr*n)/(n_attr*n);
% There are n bias values, one for each neuron (random)
initial_il_bias    = rand(1, n);
% There is n_class output, so there are total of n_class*n output weights 
% (uniform weight)
initial_ol_weights = ones(1, n_class*n)/(n_class*n);
% There are n_class bias values, one for each output neuron (random)
initial_ol_bias    = rand(1, n_class);
% starting values
starting_values = [initial_il_weights, initial_il_bias, ...
                   initial_ol_weights, initial_ol_bias];
% Setting the Genetic Algorithms tolerance for
% minimum change in fitness function before
% terminating algorithm to 1e-4 and displaying
% each iteration's results.
ga_opts = gaoptimset('TolFun',1e-4,'Display','iter', 'PlotFcn', @gaplotbestf);

% Limiting the following parameters imply 5000 max function evaluations.
% This is to maintain consistency for model comparison.
ga_opts.PopulationSize = 50;
ga_opts.Generations = 25;

% Total no. of variables in w vector
nv =  length(starting_values);

%Bounds for the weight vectors
lb=-10*ones(1,nv);
ub=10*ones(1,nv);

%Seed for randomization
setdemorandstream(100)

% Starting timer
t1 = tic;

% running the genetic algorithm with desired options
[x_opt_ga, fval, exitFlag, Output] = ga(fun,nv , [], [], [], [], lb, ub, [], ga_opts);

% Stopping timer
total_time = toc(t1);

%% BEST NETWORK PERFORMANCE
best_net = setwb(net, x_opt_ga');


trainY = round(best_net(trainX)); % Rounding converts probabilities into labels
testY = round(best_net(testX)); % Rounding converts probabilities into labels

sprintf('Total time taken - %0.2f %s', total_time, 'seconds')
sprintf('Train error - %0.4f', crossentropy(best_net, trainT, best_net(trainX), {1}))
sprintf('Test error - %0.4f', crossentropy(best_net, testT, best_net(testX), {1}))
sprintf('Train classification acc. - %0.1f%%', 100*(1-(sum((trainY(:)-trainT(:)).^2)/(2*length(trainY)))))
sprintf('Test classification acc. - %0.1f%%', 100*(1-(sum((testY(:)-testT(:)).^2)/(2*length(testY)))))