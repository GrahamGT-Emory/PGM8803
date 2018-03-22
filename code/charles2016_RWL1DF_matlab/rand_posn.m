function output = rand_posn(varargin)

% output = rand_posn(N)
%     Creates an NxN array of Poisson random variables with mean 1
% output = rand_posn(N, M)
%     Creates an MxN array of Poisson random variables with mean 1
% output = rand_posn(N, M, lambda)
%     Creates an MxN array of Poisson random variables with mean lambda
% 
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 21, 2012. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse inputs

if nargin == 1
    x_dim = varargin{1};
    y_dim = varargin{1};
    lambda_val = 1;
elseif nargin == 2
    x_dim = varargin{1};
    y_dim = varargin{2};
    lambda_val = 1;
elseif nargin == 3
    x_dim = varargin{1};
    y_dim = varargin{2};
    lambda_val = varargin{3};    
end

if lambda_val < 0
    error('mean for poisson is positive onl!')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate random variables
if max(x_dim, y_dim) == y_dim
    s1 = y_dim;
    s2 = x_dim;
    t_val = 1;
else
    s1 = x_dim;
    s2 = y_dim;
    t_val = 0;
end

output = zeros(s1, s2);
L = exp(-lambda_val);

for ii = 1:s1
    for jj = 1:s2
        temp_k = 0;
        temp_p = 1;
        while temp_p > L
            temp_k = temp_k + 1;
            temp_p = temp_p*rand(1);
        end
        output(ii, jj) = temp_k-1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign outputs

if t_val == 1
    output = output.';
else
    % do nothing
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
