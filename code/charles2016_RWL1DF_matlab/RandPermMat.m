function perm_mat = RandPermMat(varargin)

% perm_mat = RandPermMat(perm_size)
%     Generates a random permutation matrix of size perm_size with no 
%     scaling
%
% perm_mat = RandPermMat(perm_size, perm_scales)
%     Generates a random permutation matrix of size perm_size with scaling 
%     provided by perm_scales
%
% perm_mat = RandPermMat(perm_size, perm_mean, perm_var)
%     Generates a random permutation matrix of size perm_size with random 
%     scaling (Gaussian values with mean=perm_mean and variance=perm_var)
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
    perm_size = varargin{1};
    perm_scales = ones(perm_size, 1);
elseif nargin == 2
    perm_size = varargin{1};
    perm_scales = varargin{2};
    if numel(perm_scales) ~= perm_size
        error('Need one scaling per index!')
    end
elseif nargin == 3
    perm_size = varargin{1};
    perm_mean = varargin{2};
    perm_var = varargin{3};
    perm_scales = perm_mean + sqrt(perm_var)*randn(perm_size, 1);
else
    error('Bad number of inputs!')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate outputs

indx_pos = randperm(perm_size);
perm_mat = zeros(perm_size);
perm_mat((0:perm_size-1)*perm_size + indx_pos) = perm_scales;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
