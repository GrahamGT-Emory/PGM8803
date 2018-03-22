function [x_out, time_tot] = BPDN_DF_L2L1(varargin)

% [x_out, time_tot] = BPDN_DF_L2L1(y, G, F, 0.5*tau, obs_var/dyn_var, 
%                                       maxiter); 
% 
% This function solves the BPDN optimization function on a time-varying
% signal under the Gaussian innovations and sparse state assumption. Under
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
% maxiter: Maximum number of iterations for BPDN to run
% 
%    The outputs are:
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
time_bp = zeros(num_iters + 1);
in.tau = tau;
in.maxiter = maxiter;
in.record = 0;
in.x_orig = zeros(size(x_out,1),1);
in.delx_mode = 'mil';
% Initial Point
[out] = BPDN_homotopy_function(G(:, :, 1), y(:, 1), in);

x_out(:,1) = out.x_out;
time_bp(1) = out.time;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Algorithm
% Iterate through
for ii = 1:num_iters
    % Optimize the BPDN-DF objective
    [out] =...
        BPDN_homotopy_function([G(:, :, ii+1); std_quot*eye(x_dim)],...
        [y(:, ii+1); std_quot*(F(:, :, ii+1)*x_out(:, ii))], in);
    x_out(:,ii+1) = out.x_out;
    time_bp(ii+1) = out.time;
end

time_tot = sum(time_bp);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
