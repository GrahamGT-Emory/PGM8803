function [x_out, time_tot] = BPDN_DF_L1L1(varargin)

% [x_out, time_tot] = BPDN_DF_L2L1(y, G, F, tau);
% 
% This function solves the BPDN optimization function on a time-varying
% signal under the sparse innovations and sparse state assumption. Under
% these assumptions, BPDN is solved for the state at every time-step
% with the previous state used as additional measurements.
% 
%   The inputs are:
% 
% y:       MxT matrix of measurement vectors at each time step
% G:       MxNxT array of measurement matrices at each iteration
% F:       MxMxT array of dynamics matrices for each iteration
% tau:     1x2 vector containing the sparsity tradeoff parameter for BPDN 
%          wrt the innovations and the ratio of the innovations variance to
%          the measurement noise variance
% 
%    The outputs are:
% 
% x_out:      NxT matrix with the estimates of the signal x
% time_tot:   total runtime
% 
%
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 21, 2012. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input Parsing
if nargin > 1
    y = varargin{1};
    G = varargin{2};
else
    error('Bad number of inputs!')
end
if nargin > 2
    F = varargin{3};
else
    F = eye(size(G, 2));
end

if nargin > 3
    tau = varargin{4};
    if numel(tau) == 1
        tau1 = tau(1);
        tau2 = tau(1);
    elseif numel(tau) == 2
        tau1 = tau(1);
        tau2 = tau(2);
    else
        error('tau has too many elements!')
    end
else
    tau = 0.01;
end

if nargin > 4
    std_quot = varargin{5};
else
    std_quot = 1;
end

if nargin > 5
    maxiter = varargin{6};
else
    maxiter = 1000;
end

if nargin > 6
    error('Bad number of inputs!')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initializations
x_dim = size(F, 1);
num_iters = size(y, 2) - 1;

x_out = zeros(x_dim, num_iters+1);

pars.t1 = tau1;
pars.eps = 0;
pars.t2 = tau2;

options.method = 'lbfgs';
options.MaxIter = 1000;
options.MaxFunEvals = 2000;
options.Display = 'off';

tic
% Initial Point
in.tau = pars.t1;
in.maxiter = maxiter;
in.record = 0;
in.x_orig = zeros(size(x_out,1),1);
in.delx_mode = 'mil';
out = BPDN_homotopy_function(G(:, :, 1), y(:, 1), in);
x_out(:, 1) = out.x_out;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run algorithm

% Iterate through
for ii = 1:num_iters
    % Set parameters for optimization program
    out = BPDN_homotopy_function(eye(x_dim), F(:, :, ii+1)*x_out(:, ii), in);
    pars.xest = out.x_out;
    pars.G = G(:, :, ii+1);
    pars.y = y(:, ii+1);
    pars.hess = (G(:, :, ii+1).')*G(:, :, ii+1);
    % Solve BPDN-DF optimization function
    x_out(:, ii + 1) = minFunc(@opt_L1_dyn1,zeros(size(pars.G, 2), 1),options,pars);
end

time_tot = toc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
