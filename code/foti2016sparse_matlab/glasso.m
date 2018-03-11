% graphical lasso (foti et al. 2016, eq 1)
% sigma_hat - sample covariance matrix [p x p]
% lambda -  L1-norm regularization for sparsity
% S - Predicted precision matrix
% info - CVX debug information

function [S, info] = glasso(sigma_hat,lambda)

  p = size(sigma_hat);
  triu_mask = logical(ones(p)-tril(ones(p)));
  cvx_begin
    variable S(p,p) symmetric
    minimize ( -log_det(S) + trace(S*sigma_hat) + ...
      lambda*norm(S(triu_mask),1) )
  cvx_end
  
  obj = [-log_det(S), ...
    trace(S*sigma_hat), ...
    lambda*norm(S(triu_mask),1)];

  info = struct( ...
    'obj',         obj, ...
    'cvx_status',  cvx_status, ...
    'cvx_slvtol',  cvx_slvtol, ...
    'cvx_cputime', cvx_cputime);

end