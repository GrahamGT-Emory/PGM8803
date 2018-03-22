function [f_val, del_x, hess_x] = opt_L1_dyn1(x_pt, pars)


% [f_val, del_x, hess_x] = opt_l1_fun(x_pt, pars)
% 
% This function calculates the value of the optimization function, its
% derivative and its hessian. The optimization function includes an
% ell_2 norm over a set of measurements, an ell_1 norm over the value of
% the function's variable and an ell_1 norm of the difference between the
% function's argument and a prediction:
%
% f(x) = ||y - G*x||_2^2 + pars.t1*||x||_1 + pars.t2*||x - pars.xest||_1
% 
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 14, 2012. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err = pars.y - pars.G*x_pt;
err_val = sum(err.^2) - pars.eps;


f_val = err_val + pars.eps + pars.t1*sum(abs(x_pt))...
    + pars.t2*sum(abs(x_pt - pars.xest));

del_x = pars.t1*sign(x_pt) + pars.t2*sign(x_pt - pars.xest) - 2*(pars.G.')*err;

hess_x =  2*pars.hess; 


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
