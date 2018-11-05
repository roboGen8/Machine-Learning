%% INITIALIZING NEURAL NETWORK
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

%% HILL CLIMBING OPTIMIZATION
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
%starting_values = rand(1, 2777);
%%%%% Optimizing using patternsearch 
% Add 'Display' option to display result of iterations
ps_opts = psoptimset ( 'CompletePoll', 'off', 'Display', 'iter', 'PlotFcns', {@psplotbestf},...
     'Cache', 'On', 'PollingOrder', 'random'); %, 'TimeLimit', 120 );
ps_opts.MaxFunEvals = 5000;
ps_opts.TolFun = 1e-4;
ps_opts.MaxIter = 80;
%ps_opts = psoptimset ( 'CompletePoll', 'off', 'Display', 'iter', 'MaxIter', 1000, 'PlotFcn', {@psplotbestf},...
%     'MaxMeshSize', 108, 'TolX', 1e-1, 'Cache', 'On', 'PollingOrder', 'random', 'MaxFunEvals', 1000); %, 'TimeLimit', 120 );

% Seed for randomization
setdemorandstream(100)

nv =  length(starting_values);
lb=-inf*ones(1,nv);
ub=inf*ones(1,nv);

mnn = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25];
best_err = ones(1,11);
best_err1 = ones(1,11);
for i = 1:11
    % Initialize neural network
    net = patternnet(n);

    % Configure the neural network for this dataset
    net = configure(net, x, t); 

    % Seed for randomization
    setdemorandstream(100)
    mn = mnn(i);
    starting_values = randn(1,nv) + mn;
    
    [x_opt_hill, fval, flag, output] = patternsearch(fun, starting_values, [], [],[],[], lb, ub, ps_opts);
    
    best_net = setwb(net, x_opt_hill');
    best_err(i) = crossentropy(best_net, trainT, best_net(trainX), {1});
    best_err1(i) = crossentropy(best_net, testT, best_net(testX), {1});
end

plot(mnn, best_err);



%% BEST NETWORK PERFORMANCE
% Initialize neural network
net = patternnet(n);

% Configure the neural network for this dataset
net = configure(net, x, t); 

% Seed for randomization
setdemorandstream(100)

best_starting_values = randn(1,nv) -25;
t1 = tic;
[x_opt_hill, fval, flag, output] = patternsearch(fun, best_starting_values, [], [],[],[], lb, ub, ps_opts);
total_time = toc(t1);
best_net = setwb(net, x_opt_hill');


trainY = round(best_net(trainX)); % Rounding converts probabilities into labels
testY = round(best_net(testX)); % Rounding converts probabilities into labels

sprintf('Total time taken - %0.2f %s', total_time, 'seconds')
sprintf('Train error - %0.4f', crossentropy(best_net, trainT, best_net(trainX), {1}))
sprintf('Test error - %0.4f', crossentropy(best_net, testT, best_net(testX), {1}))
sprintf('Train classification acc. - %0.1f%%', 100*(1-(sum((trainY(:)-trainT(:)).^2)/(2*length(trainY)))))
sprintf('Test classification acc. - %0.1f%%', 100*(1-(sum((testY(:)-testT(:)).^2)/(2*length(testY)))))