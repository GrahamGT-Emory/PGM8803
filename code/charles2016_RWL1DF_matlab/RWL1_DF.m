function [x_est, lambda_est, time_tot] = RWL1_DF(varargin)

% [x_est, lambda_est, time_tot] = RWL1_DF(y_meas, G_n, F_n, ...
%           dyn_var, obs_var, num_EM, tau, beta)
% 
% Small scale implementation of the RWL1-DF algorithm. Accepts minimum 3,
% maximum 8 inputs.
% 
%   The inputs are:
% 
% y_meas:  MxT matrix of measurement vectors at each time step
% G_n:     MxNxT array of measurement matrices at each iteration
% F_n:     MxMxT array of dynamics matrices for each iteration
% dyn_var: Scalar value of the expected signal variance (sets initial
%          variance and beta = sqrt(dyn_var) if unspecified. (default =
%          0.01)
% obs_var: Observation variance. (default = 0.01)
% num_EM:  Number of Expectation-Maximization iterations to run (default -
%          10)
% tau:     lambda_0 - baseline regularization parameter for BPDN (default =
%          sqrt(obs_var)) 
% beta:    RW denominator regulation parameter. (default = sqrt(dyn_var))
% 
%    The outputs are:
% x_est:      NxT matrix with the estimates of the signal x
% lambda_est: NxT matrix with the estimates of the variances lambda
% time_tot:   total runtime
% 
% This function takes the set of measurements y over time and causally
% estimates a set of generative signals x which were measured by y as
%              y[t] = G[t]*x[t] + e[t]
% The signal x is assumed to have some approximately known dynamics
%              x[t] = F[t]*x[t-1] + v[t]
% The estimate at each time step is evaluated as an EM iteration:
%      E-Step: update lambda:
%              lambda[t] <-- 2/(|x[t]| + |F[t]*x[t-1]| + beta)
%      M-step: do weighted BPDN:
%              x[t] <-- argmin{ ||y[t] - G[t]*x||_2^2 + tau*|L*x||_1
% where L = diag(lambda[t]). This code assumes x is canonically sparse. If
% not, a synthesis model can be worked in by setting
%              G <-- G*W
%              F <-- W^T*F*W
% Where W is the synthesis basis. then multiply W*x_est to recover the
%     non-canonical sparse values.
%
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 20, 2012. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse and test inputs

if nargin >= 3
    y_meas = varargin{1};
    G_n = varargin{2};
    F_n = varargin{3};
else
    error('Too few inputs!')
end
if nargin > 3
    dyn_var = varargin{4};
else
    dyn_var = 0.01;
end
if nargin > 4
    obs_var = varargin{5};
else
    obs_var = 0.01;
end
if nargin > 5
    num_EM = varargin{6};
else
    num_EM = 10;
end

if nargin > 6
    tau = varargin{7};
else
    tau = (obs_var);
end

if nargin > 7
    beta = varargin{8};
else
    beta = sqrt(dyn_var);
end

if nargin > 8
    warning('Input:Error', 'Too many inputs! Ignoring the extra inputs...')
end

% Check dynamic matrix size
if (size(F_n, 2) ~= size(F_n, 1))
    error('F_n should be square!')
end

% Check measurement size
if size(G_n, 1) ~= size(y_meas, 1)
    error('G_n should map to the same dimention space as y is in!')
end

% Check if number of iterations is consistant
if (size(G_n, 3) ~= size(y_meas, 2))||(size(G_n, 3) ~= size(y_meas, 2))
    error('The number of iterations should be consistant!')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initializations and parsing

dim_x = size(F_n, 1);       % Dimension of the signal x
num_iter = size(y_meas, 2); % Number of time-steps
maxiter = 2000;             % Number of BPDN (NOT RWL1!!) iterations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run Algorithm

tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% First Time-step
% Initialize lambda and the signal predictions
lambda_est = ones(dim_x, num_iter).*(dyn_var);
x_est = zeros(dim_x, num_iter);
lambda_pred = ones(dim_x, 1).*(obs_var);

% Initalize the estimates of x and lambda for the current time-step
in.tau = tau;
in.maxiter = maxiter;
in.record = 0;
in.x_orig = zeros(dim_x,1);
in.delx_mode = 'mil';

lambda_EM = zeros(dim_x, 1);
x_EM = zeros(dim_x, 1);

for index_EM = 1:num_EM;
    % E-step: Update lambda
    lambda_EM = 2./(1./lambda_pred + abs(x_EM));

    % M-step: Solve weighted BPDN
    [out] = BPDN_homotopy_function(G_n(:, :, 1)*diag(1./lambda_EM), ...
        y_meas(:, 1), in);
    
    x_EM = out.x_out./lambda_EM;
end

% Store first time-step's results
lambda_est(:, 1) = lambda_EM;
x_est(:, 1) = x_EM;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Remaining Time-steps

for index = 2:num_iter
    % Predict Lambda (i.e. Predict 1/x + e)
    lambda_pred = 1./(abs(F_n(:, :, index)*x_est(:, index-1)) + beta);
    % Initalize current estimates of x and lambda
    lambda_EM = zeros(dim_x, 1);
    x_EM = zeros(dim_x, 1);
    for index_EM = 1:num_EM;
        % E-step: Update lambda
        lambda_EM = 2./(1./lambda_pred + abs(x_EM));
        
        % M-step: Solve weighted BPDN
        [out] = BPDN_homotopy_function(G_n(:, :, index)*diag(1./lambda_EM), ...
        y_meas(:, index), in);
        x_EM = out.x_out./lambda_EM;
    end
    
    lambda_est(:, index) = lambda_EM;
    x_est(:, index) = x_EM;
end
% Save runtime for comparison
time_tot = toc;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
