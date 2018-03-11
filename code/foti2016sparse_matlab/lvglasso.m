% sparse + low rank graphical lasso (ma et. al 2013, [eq 1.2])
% C - sample covariance matrix [p x p]
% lambda -  L1-norm regularization for sparsity
% S - Predicted observed precision matrix
% L - Predicted hidden precision matrix
% info - CVX debug information

function [S, L , info] = lvglasso(C, alpha, beta)

  p = size(C);
  triu_mask = logical(ones(p)-tril(ones(p)));
  cvx_begin 
    variable S(p,p) complex symmetric semidefinite
    variable L(p,p) complex symmetric semidefinite 
    minimize ( trace((S - L) * C) - log_det(S - L) ...
      + alpha*norm(S(triu_mask),1) + beta*trace(L))  
  cvx_end
  
  obj = [trace((S - L) * C) - log_det(S - L) ...
      + alpha*norm(S(triu_mask),1) + beta*trace(L)];

  info = struct( ...
    'obj',         obj, ...
    'cvx_status',  cvx_status, ...
    'cvx_slvtol',  cvx_slvtol, ...
    'cvx_cputime', cvx_cputime);

end