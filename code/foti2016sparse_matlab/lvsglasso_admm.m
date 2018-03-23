% LVSGLASSO sparse + low-rank time series graphical lasso (foti et al 2016)
%           (solve using ADMM updates in appendix A)
% INPUTS
%  Psi         : spectral density matrix [p x p x F]
%  lambdaPsi   : trade-off parameter for l1 norm
%  lambdaL     : trade-off parameter for trace(L)
%  debugPlot   : if true, plot debug info at each iteration
% OUTPUTS
%  S           : recovered sparse component
%  L           : recevored low-rank matrix
%  info        : debug information
%
% See Foti et al. 2016, appendix A:
%  R = Psi - L     (p x p x F)
%  Z = [R, Psi, L]   (p x 3p x F)
%  Ztilde = [Rtilde, Psitilde, Ltilde]  (p x 3p x F)


function [out] = lvsglasso_admm(Shat,lambdaPsi,lambdaL,opts)

% ***Pay careful attention to lambdaPsi (regularization) vs. LambdaPsi
% (dual variable)***
  


  
  % --- initialize ---
  % --- user defined parameters ---
  if nargin < 4
      maxiter = 20;
      mu = 1; % TODO
      mu_dim_factor = 0.001; % TODO
      debugPlot = false;
  else
      maxiter = opts.maxiter;
      mu = opts.mu; % TODO
      mu_dim_factor = opts.mu_dim_factor; % TODO
      debugPlot = opts.debugPlot;     
  end
  ABSTOL   = 1e-5;
  RELTOL   = 1e-5;
  [p,~,F] = size(Shat);
  pos = @(X) double(X>0) .* X;
  normsAdotij = @(A) sqrt(sum(A,3));
  terminate = false;
  iter = 0;
  
  R_k = repmat(eye(p), 1, 1, F);
  Psi_k = Shat;
  L_k = zeros(p,p,F);
  
  Rtilde_k = R_k;
  Psitilde_k  = Psi_k;
  Ltilde_k    = L_k;
  
  LambdaR_k = zeros(p,p,F);
  LambdaPsi_k = zeros(p,p,F);
  LambdaL_k = zeros(p,p,F);
  Lambda_k = zeros(p,3*p,F);
  Z_k = zeros(p,3*p,F);
  Ztilde_k = zeros(p,3*p,F);
  eigR_k = zeros(p,F);

  while ~terminate


%     fprintf('LVSGLASSO-ADMM: iteration %d of %d...\n', iter+1, maxiter);
    
    
    % --- update primals (Z = [R[.], Psi[.], L[.]]) ---

    % (22): update R
    R_k1 = zeros(p,p,F);
    for f = 1:F
      tmp = mu*Shat(:,:,f) - Rtilde_k(:,:,f) - mu*LambdaR_k(:,:,f); tmp = (tmp + tmp') / 2;
      [U,V] = eig(tmp);
      v = diag(V);
      Vbar = 1/2*diag(-v + sqrt(v.^2 + 4/mu));
      tmp = U*Vbar*U'; tmp = (tmp + tmp') / 2;
      R_k1(:,:,f) = tmp;
      eigR_k(:, f) = diag(Vbar);
    end
    
    % (23): update L
    L_k1 = zeros(p,p,F);
    
    for f = 1:F
      tmp = Ltilde_k(:,:,f) + mu*LambdaL_k(:,:,f); tmp = (tmp + tmp.') / 2; 
      [U,V] = eig(tmp);
      v = diag(V);
%       definition according to foti et al 2016 ref [7]
%       vbar = soft_shrink(v, lambdaL*mu);
      vbar = max(v - lambdaL*mu, 0);
      Vbar = diag(vbar);
      tmp = U*Vbar*U.'; tmp = (tmp + tmp') / 2; 
      L_k1(:,:,f) = tmp;
    end
    
    % (24): update Psi - foti et al 2016 ref [7] - eq 3.16
    Psi_k1 = zeros(p,p,F);
    A = Psi_k + mu*LambdaPsi_k;
    
    for f = 1:F
      % For off-diagonal entries   
      Psi_k1(:,:,f) = pos(1-((lambdaPsi*mu)./normsAdotij(A))).* A(:,:,f);
    end
    % For on-diagonal entries 
    Psi_k1(logical(repmat(eye(p), 1, 1, F))) =  A(logical(repmat(eye(p), 1, 1, F)));

    % --- update duals (Ztilde = [Rtilde[.], PsiTilde[.], Ltilde[.]]) ---

    % (25-27): calculate Rbar, PsiBar, Lbar
    Rbar_k = R_k1 - mu*LambdaR_k;
    Psibar_k = Psi_k1 - mu*LambdaPsi_k;
    Lbar_k = L_k1 - mu*LambdaL_k;

    Gamma = -(Rbar_k - Psibar_k + Lbar_k) / 3;
    % (28): update Rtilde
    Rtilde_k1 = Rbar_k + Gamma ;
    % (29): update Psitilde
    Psitilde_k1 = Psibar_k - Gamma;
    % (30): update Ltilde
    Ltilde_k1 = Lbar_k + Gamma;


    % --- update splitting variable (Lambda) ---
    Z_k1 = [R_k1, Psi_k1, L_k1];
    Ztilde_k1    = [Rtilde_k1, Psitilde_k1, Ltilde_k1];
    Lambda_k1    = Lambda_k - ((Z_k1 - Ztilde_k1) / mu);
    LambdaR_k1   = Lambda_k1(:,1:p,:);
    LambdaPsi_k1 = Lambda_k1(:,p+1:2*p,:);
    LambdaL_k1   = Lambda_k1(:,2*p+1:end,:);


    % --- check convergence ---
    iter = iter + 1;
    % update mu
    mu = mu - mu_dim_factor; % see foti et al. 2016 ref [15]
    % calculate primal/dual residuals
    r_k = sum(R_k - Psi_k + L_k,3);
    s_k = Ztilde_k1 - Ztilde_k;
    % (15) update epsilon
    Z_k1_norms = sqrt(sum(sum(Z_k.^2,1),2)); % TODO Z_k or Z_k1?
    Ztilde_k1_norms = sqrt(sum(sum(Ztilde_k1.^2,1),2)); % TODO Ztilde_k or Ztilde_k1?
    Lambda_k1_norms = sqrt(sum(sum(Lambda_k1.^2,1),2)); % TODO Lambda_k or Lambda_k1
    epri = sqrt(p^2*F)*ABSTOL + RELTOL*sum(max(Z_k1_norms,Ztilde_k1_norms));
    edual = sqrt(p^2*F)*ABSTOL + RELTOL*sum(Lambda_k1_norms);
    % check convergence
    terminate = iter > maxiter || (norm(r_k(:)) < epri && norm(s_k(:)) < edual);
    history.eps_pri(iter) =  epri;
    history.eps_dual(iter) =  edual;
    history.r_norm(iter) =  norm(r_k(:));
    history.s_norm(iter) =  norm(s_k(:));
    history.objval(iter) =  objective(R_k, Shat, eigR_k, Psi_k, L_k,lambdaPsi, lambdaL);
    % --- debug plot ---
    if debugPlot
      clf;
      subplot(211); imagesc(abs(Psi_k(:,:,floor(F/2)))); axis image;
      title(sprintf('Iteration %d',iter));
      subplot(212); imagesc(abs(L_k(:,:,floor(F/2)))); axis image;
      drawnow;
    end
    
    
    % --- update for next iteration ---
    R_k         = R_k1;
    Psi_k       = Psi_k1;
    L_k         = L_k1;
    Rtilde_k    = Rtilde_k1;
    Psitilde_k  = Psitilde_k1;
    Ltilde_k    = Ltilde_k1;
    Z_k         = Z_k1;
    Lambda_k    = Lambda_k1;
    LambdaR_k   = LambdaR_k1;
    LambdaPsi_k = LambdaPsi_k1;
    LambdaL_k   = LambdaL_k1;
    

  end

  
  % --- outputs ---   
  out.L = L_k;
  out.S = Psi_k;
  out.obj = objective(R_k,Shat,eigR_k,Psi_k, L_k,lambdaPsi, lambdaL);
  out.history = history;
end

function obj = objective(R,Shat,eigR,Psi, L,lambdaPsi, lambdaL)
obj = 0; 
for ii = 1:size(R, 3)
    obj = obj + sum(sum(R(:,:,ii).*Shat(:,:,ii))) - sum(log(eigR(:,ii))) ...
        + lambdaPsi*sum(sum(abs(Psi(:,:,ii)))) + lambdaL*trace(L(:,:,ii));
end
end

function x = soft_shrink(z,tau)
x = sign(z).*max(abs(z)-tau,0);
end