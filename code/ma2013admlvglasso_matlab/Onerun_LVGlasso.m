clc; clear all; close all;
%% load data
dataformat = 'rand-1'; alpha = 0.05; beta = 0.25; 
input_dim = 100; SigmaO = getdata(dataformat,input_dim); n = size(SigmaO,1);

do_ADMM_B = 1; do_ADMM_R = 0; 
opts.continuation = 1; opts.mu = n; opts.num_continuation = 10; opts.eta = 1/4; opts.muf = 1e-6;
opts.maxiter = 500; opts.stoptol = 1e-5; 

if do_ADMM_B
    opts.over_relax_par = 1.6; 
    tic; out_B = ADMM_B(SigmaO,alpha,beta,opts); solve_B = toc;
    fprintf('ADMM_B: obj: %e, iter: %d, cpu: %3.1f \n',out_B.obj,out_B.iter,solve_B); 
end
if do_ADMM_R
    opts.tau = 0.6; 
    tic; out_R = ADMM_R(SigmaO,alpha,beta,opts); solve_R = toc; 
    fprintf('ADMM_R: obj: %e, iter: %d, cpu: %3.1f, resid:%e \n',out_R.obj,out_R.iter,solve_R,out_R.resid); 
end
