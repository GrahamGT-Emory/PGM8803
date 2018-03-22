function [x, y, G, F] = rand_sparse_seq(varargin)

% [x, y, G, F] = rand_sparse_seq(x_dim, y_dim, sparse_num, init_var, ...
%          num_iters, obs_var, dyn_var, poi_var)
%
% This function generates a sequence of sparse signals evolving with random 
% permutations at each iteration. The innovations are sparse. Each signal
% is compressed with a Gaussian matrix to yield compressed measurements.
% 
%   The inputs are:
% 
% x_dim:      scalar value indicating the size of the signal
% y_dim:      scalar value indicating the number of measurements
% sparse_num: scalar value indicating the signal sparsity
% init_var:   Wavelet transform (sparsity basis)
% num_iters:  scalar value for the number of time iterations to generate
% obs_var:    scalar value for the variance of the measurement noise
% dyn_var:    scalar value for the on-support variance of dynamics noise
% poi_var:    scalar value for the innovations sparsity Poisson mean
% 
%    The outputs are:
% 
% x:  NxT array of dynamically evolving sparse signals
% y:  MxT array of the compressed measurements of x
% G:  MxNxT array of measurement matrices that gave y from x
% F:  NxNxT array of dynamics matrices that approximately evolve x
% 
% 
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 20, 2012. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse inputs

if nargin >= 2
    x_dim = varargin{1};
    y_dim = varargin{2};
else
    x_dim = 500;
    y_dim = 70;
end

if nargin > 2
    sparse_num = varargin{3};
else
    sparse_num = 20;
end

if nargin > 3
    init_var = varargin{4};
else
    init_var = 1;
end

if nargin > 4
    num_iters = varargin{5};
else
    num_iters = 30;
end

if nargin > 5
    obs_var = varargin{6};
else
    obs_var = 0.1;
end

if nargin > 6
    dyn_var = varargin{7};
else
    dyn_var = 0.1;
end

if nargin > 7
    poi_var = varargin{8};
else
    poi_var = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize the sequence

x = zeros(x_dim, num_iters+1);
y = zeros(y_dim, num_iters+1);
F = zeros(x_dim, x_dim, num_iters+1);
G = zeros(y_dim, x_dim, num_iters+1);

init_supp = randperm(x_dim);
x(init_supp(1:sparse_num), 1) = sqrt(init_var)*randn(sparse_num, 1);

G(:, :, 1) = randn(y_dim, x_dim)/sqrt(y_dim);
y(:, 1) = G(:, :, 1)*x(:, 1) + sqrt(obs_var)*randn(y_dim, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate the sequence

for ii = 2:num_iters
    F(:, :, ii) = RandPermMat(x_dim, 1, 0.01);
    x(:, ii) = noise_model(F(:, :, ii)*x(:, ii-1), dyn_var, poi_var);
    G(:, :, ii) = randn(y_dim, x_dim)/sqrt(y_dim);
    y(:, ii) = G(:, :, ii)*x(:, ii) + sqrt(obs_var)*randn(y_dim, 1);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

