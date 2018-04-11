function out = ADMM_GLASSO(SigmaO,alpha,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file implements the Alternating Direction Methods for 
% Gaussian Graphical Model Selection  
% min <R,SigmaO> - logdet(R) + alpha ||S||_1  
% s.t. R positive definite and R - S = 0
%
% Based off code from 
% "Alternating Direction Methods for Latent Variable Gaussian Graphical
% Model Selection", appeared in Neural Computation, 2013, 
% by Ma, Xue and Zou, for solving 
% Latent Variable Gaussian Graphical Model Selection 
%
% Date: Mar 20, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
QUIET    = 1;
ABSTOL   = 1e-5;
RELTOL   = 1e-5;
n = size(SigmaO,1); R = eye(n,n); S = R;
RY = R; SY = S;  Y = [RY;SY];  
Lambda1 = zeros(size(R)); Lambda2 = Lambda1; 
mu = opts.mu; eta = opts.eta; over_relax_par = opts.over_relax_par;

for iter = 1: opts.maxiter
    % update X = (R,S)
    B1 = RY + mu*Lambda1;
    B2 = SY + mu*Lambda2;
    tmp = mu*SigmaO-B1; tmp = (tmp+tmp')/2;
%     [U,D] = mexeig(tmp); d = diag(D);
    [U,D] = eig(tmp); d = diag(D);
    eigR = (-d + sqrt(d.^2+4*mu))/2;
    R = U*diag(eigR)*U'; R = (R+R')/2;
    
    S = soft_shrink(B2,alpha*mu); S = (S+S')/2;
    
    
    X = [R;S]; 
    RO = over_relax_par*R + (1-over_relax_par)*RY;
    SO = over_relax_par*S + (1-over_relax_par)*SY;
    % update Y = (RY,SY)
    
    Y_old = Y; 
    B1 = RO - mu*Lambda1;
    B2 = SO - mu*Lambda2;
    Gamma = -(B1-B2)/2;
    RY = B1 + Gamma; 
    SY = B2 - Gamma; 
    Y = [RY;SY];
    % update Lambda
    Lambda1 = Lambda1 - (RO-RY)/mu; Lambda1 = (Lambda1 + Lambda1')/2;
    Lambda2 = Lambda2 - (SO-SY)/mu; Lambda2 = (Lambda2 + Lambda2')/2;
    Lambda = [Lambda1;Lambda2];
    % diagnostics, reporting, termination checks
    k = iter; 
    history.objval(k)  = objective(R,SigmaO,eigR,S,alpha); 

    history.r_norm(k)  = norm(X - Y,'fro');
    history.s_norm(k)  = norm(-(Y - Y_old)/mu);

    history.eps_pri(k) = sqrt(2*n*n)*ABSTOL + RELTOL*max(norm(X,'fro'), norm(Y,'fro'));
    history.eps_dual(k)= sqrt(2*n*n)*ABSTOL + RELTOL*norm(Lambda,'fro');

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    % print stats 
    resid = norm(R-S,'fro')/max([1,norm(R,'fro'),norm(S,'fro')]);
 
    obj = history.objval(k);  
    if opts.continuation && mod(iter,opts.num_continuation)==0; mu = max(mu*eta,opts.muf); end;
     
end
out.R = R; out.S = S;  out.obj = obj; out.eigR = eigR; out.resid = resid; out.iter = iter;


function x = soft_shrink(z,tau)
x = sign(z).*max(abs(z)-tau,0);

function obj = objective(R,SigmaO,eigR,S,alpha)
obj = sum(sum(R.*SigmaO)) - sum(log(eigR)) + alpha*sum(abs(S(:)));
