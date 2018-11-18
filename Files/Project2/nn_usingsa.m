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
fun = @(w) ce_test(w, net, trainX, trainT);

% Unbounded
lb = -inf;
ub = inf;

% Add 'Display' option to display result of iterations
% Adding a function tolerance of 1e-4 (one of stopping criteria)
sa_opts = saoptimset('TolFun', 1e-4, 'Display', 'iter', 'PlotFcns', {@saplotbestf, @saplotf});
sa_opts.MaxFunEvals = 100; % For consistency across solvers
sa_opts.InitialTemperature = 5;
sa_opts.AnnealingFcn = @annealingboltz;

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
% Starting values
starting_values = [initial_il_weights, initial_il_bias, ...
                   initial_ol_weights, initial_ol_bias];
               
% Seed for randomization
setdemorandstream(100)

% Random starting point
starting_values = rand(1, length(starting_values));

t1 = tic;
[x_opt_sa, fval, flag, output] = simulannealbnd(fun, starting_values, lb, ub, sa_opts);
total_time = toc(t1);

%% BEST NETWORK PERFORMANCE
best_net = setwb(net, x_opt_sa');


trainY = round(best_net(trainX)); % Rounding converts probabilities into labels
testY = round(best_net(testX)); % Rounding converts probabilities into labels

sprintf('Total time taken - %0.2f %s', total_time, 'seconds')
sprintf('Train error - %0.4f', crossentropy(best_net, trainT, best_net(trainX), {1}))
sprintf('Test error - %0.4f', crossentropy(best_net, testT, best_net(testX), {1}))
sprintf('Train classification acc. - %0.1f%%', 100*(1-(sum((trainY(:)-trainT(:)).^2)/(2*length(trainY)))))
sprintf('Test classification acc. - %0.1f%%', 100*(1-(sum((testY(:)-testT(:)).^2)/(2*length(testY)))))