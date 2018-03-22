function [x_out, time_tot] = BPDN_multi(varargin)

% [x_out, time_tot] = BPDN_multi(y, G, tau, 1000);
% 
% This function solves the BPDN optimization function on a time-varying
% signal by solving for each time-step independently.
% 
%   The inputs are:
% 
% y:       MxT matrix of measurement vectors at each time step
% G:       MxNxT array of measurement matrices at each iteration
% tau:     1x2 vector containing the sparsity tradeoff parameter for BPDN 
%          wrt the innovations and the ratio of the innovations variance to
%          the measurement noise variance
% maxiter: The maximum number of iterations for the BPDN solver
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
% Last updated August 14, 2012. 
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
    tau = varargin{3};
else
    tau = 0.01;
end

if nargin > 3
    maxiter = varargin{4};
else
    maxiter = 1000;
end
if nargin > 4
    error('Bad number of inputs!')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initializations
x_dim = size(G, 2);
num_iters = size(y, 2) - 1;
in.tau = tau;
in.maxiter = maxiter;
in.record = 0;
in.x_orig = zeros(x_dim,1);
in.delx_mode = 'mil';

x_out = zeros(x_dim, num_iters+1);
time_bp = zeros(num_iters + 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve all BPDN optimization functions
% Can do this in parallel with parfor
parfor ii = 1:num_iters+1
    % Optimize BPDN objective
    [out] = BPDN_homotopy_function(G(:, :, ii),...
        y(:, ii), in);
    x_out(:,ii) = out.x_out;
    time_bp(ii) = out.time;    
end

time_tot = sum(time_bp);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
