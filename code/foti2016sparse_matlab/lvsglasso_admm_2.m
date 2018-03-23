function out = lvsglasso_admm_2(SigmaO,alpha,beta,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file implements the Alternating Direction Methods for 
% Latent Variable Spectral Gaussian Graphical Model Selection  
% min <R,SigmaO> - logdet(R) + alpha ||S||_1  
% s.t. R positive definite and R - S = 0
%
% Based off code from 
% "Alternating Direction Methods for Latent Variable Gaussian Graphical
% Model Selection", appeared in Neural Computation, 2013, 
% by Ma, Xue and Zou, for solving 
% Latent Variable Gaussian Graphical Model Selection 
%
% Date: Mar 22, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
QUIET    = 1;
ABSTOL   = 1e-5;
RELTOL   = 1e-5;
[n, ~, F] = size(SigmaO); R = repmat(eye(n), 1, 1, F); S = R; L = zeros(n,n,F);
RY = R; SY = S; LY = L; Y = [RY;SY;LY];  
Lambda1 = zeros(size(R)); Lambda2 = Lambda1; Lambda3 = Lambda1; 
mu = opts.mu; eta = opts.eta; over_relax_par = opts.over_relax_par;
normsAdotij = @(A) sqrt(sum(A,3));
frob_norm = @(X)  sqrt(sum(sum(sum(X.^2))));
for iter = 1: opts.maxiter
    % update X = (R,S,L)
    B1 = RY + mu*Lambda1;
    B2 = SY + mu*Lambda2;
    B3 = LY + mu*Lambda3; 
    tmp = mu*SigmaO-B1; 
    % Pre-process S  % TODO - should do soft thresholding?
    S = soft_shrink(B2,alpha*mu);    
    for f = 1:F
        % Update R
        tmp_f = (tmp(:,:,f)+tmp(:,:,f)')/2;
        [U,D] = eig(tmp_f); d = diag(D);
        eigR_f = (-d + sqrt(d.^2+4*mu))/2;
        eigR(:,f) = eigR_f;
        R(:,:,f) = U*diag(eigR_f)*U'; R(:,:,f) = (R(:,:,f)+R(:,:,f)')/2;
        
        % Update L
        B3(:,:,f) = (B3(:,:,f)+B3(:,:,f)')/2;
        [U,D] = eig(B3(:,:,f)); d = diag(D);
        eigL_f = max(d-mu*beta,0);
        eigL(:,f) = eigL_f;
        L(:,:,f) = U*diag(eigL_f)*U'; L(:,:,f)  = (L(:,:,f) +L(:,:,f)')/2;
       
        S(:,:,f) = (S(:,:,f)+S(:,:,f)')/2;
    end
    

    
    % Update S
    % Group lasso Penalty
    A = S + mu*Lambda2;
    A_norms = normsAdotij(A);
    A_norms(A_norms == 0) = 1;
    A_norms = real(A_norms);
    for f = 1:F 
      A_f = A(:,:,f);
      % For off-diagonal entries   
      S(:,:,f) = pos(1-((alpha*mu)./A_norms)).* A_f;  
    end
    % For on-diagonal entries 
    S(logical(repmat(eye(n), 1, 1, F))) =  A(logical(repmat(eye(n), 1, 1, F)));

    X = [R;S;L]; 
    RO = over_relax_par*R + (1-over_relax_par)*RY;
    SO = over_relax_par*S + (1-over_relax_par)*SY;
    LO = over_relax_par*L + (1-over_relax_par)*LY;
    % update Y = (RY,SY,LY)
    
    Y_old = Y; 
    B1 = RO - mu*Lambda1;
    B2 = SO - mu*Lambda2;
    B3 = LO - mu*Lambda3; 
    Gamma = -(B1-B2+B3)/3;
    RY = B1 + Gamma; 
    SY = B2 - Gamma; 
    LY = B3 + Gamma; 
    Y = [RY;SY;LY];
    % update Lambda
    Lambda1 = Lambda1 - (RO-RY)/mu; 
    Lambda2 = Lambda2 - (SO-SY)/mu; 
    Lambda3 = Lambda3 - (LO-LY)/mu; 
    for f = 1:F
        Lambda1(:,:,f) = (Lambda1(:,:,f) + Lambda1(:,:,f)')/2;
        Lambda2(:,:,f) = (Lambda2(:,:,f) + Lambda2(:,:,f)')/2;
        Lambda3(:,:,f) = (Lambda3(:,:,f) + Lambda3(:,:,f)')/2;
    end
    Lambda = [Lambda1;Lambda2;Lambda3];
    % diagnostics, reporting, termination checks
    k = iter; 
    history.objval(k)  = objective(R,SigmaO,eigR,S,eigL,alpha,beta); 
    
    r_k = sum(R - S + L,3);
    s_k = Y - Y_old;
    history.r_norm(k)  = norm(r_k(:));
    history.s_norm(k)  = norm(s_k(:));
    
    history.eps_pri(k) = sqrt(n^2*F)*ABSTOL + RELTOL*max(frob_norm(X), frob_norm(Y));
    history.eps_dual(k)= sqrt(n^2*F)*ABSTOL + RELTOL*frob_norm(Lambda);

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
    resid = frob_norm(R-S+L)/max([1,frob_norm(R),frob_norm(S),frob_norm(L)]);
 
    obj = history.objval(k);  
    if opts.continuation && mod(iter,opts.num_continuation)==0; mu = max(mu*eta,opts.muf); end;
     
end
out.R = R; out.S = S; out.L = L; out.obj = obj; out.eigR = eigR; out.eigL = eigL; out.resid = resid; out.iter = iter;
end

function x = soft_shrink(z,tau)
x = sign(z).*max(abs(z)-tau,0);
end

function obj = objective(R,SigmaO,eigR,S,eigL,alpha,beta)
obj = 0; 
for ii = 1:size(R, 3)
    obj = obj + sum(sum(R(:,:,ii).*SigmaO(:,:,ii))) - sum(log(eigR(:,ii))) ...
        + alpha*sum(sum(abs(S(:,:,ii)))) + beta*sum(eigL(:,ii));
end
end
