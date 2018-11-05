function ce_calc = ce_test(x, net, inputs, targets)
% 'x' contains the weights and biases vector
% in row vector form. This must be transposed
% when being set as the weights and biases
% vector for the network.

% To set the weights and biases vector to the
% one given as input
net = setwb(net, x');

% To evaluate the ouputs based on the given
% weights and biases vector
y = net(inputs);

% Calculating the cross-entropy error
ce_calc = crossentropy(net, targets, y, {1});
 

end