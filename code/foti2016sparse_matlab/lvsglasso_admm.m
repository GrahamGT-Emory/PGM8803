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


function [S,L,info] = lvsglasso_admm(Psi,lambdaPsi,lambdaL,debugPlot)

  
  % --- user defined parameters ---
  maxiter = 20;
  mu = 1; % TODO
  mu_dim_factor = 0.001; % TODO

  
  % --- initialize ---
  if nargin < 4
    debugPlot = false;
  end
  [p,~,F] = size(Psi);
  pos = @(X) double(X>0) .* X;
  normsAdotij = @(A) sqrt(sum(A,3));
  terminate = false;
  i = 0;
  Psi_k = Psi;
  R_k = zeros(p,p,F);
  L_k = zeros(p,p,F);
  Rtilde_k = zeros(p,p,F);
  LambdaR_k = zeros(p,p,F);
  LambdaPsi_k = zeros(p,p,F);
  LambdaL_k = zeros(p,p,F);
  Lambda_k = zeros(p,3*p,F);
  Z_k = zeros(p,3*p,F);
  Ztilde_k = zeros(p,3*p,F);


  while ~terminate


    fprintf('LVSGLASSO-ADMM: iteration %d of %d...\n', i+1, maxiter);
    
    
    % --- update primals (Z = [R[.], Psi[.], L[.]]) ---

    % (22): update R
    R_k1 = zeros(p,p,F);
    for f = 1:F
      [U,V] = eig(mu*Psi_k(:,:,f) - Rtilde_k(:,:,f) - mu*LambdaR_k(:,:,f));
      v = diag(V);
      Vbar = 1/2*diag(-v + sqrt(v.^2 + 4/mu));
      R_k1(:,:,f) = U*Vbar*U.';
    end
    % (23): update L
    L_k1 = zeros(p,p,F);
    for f = 1:F
      [U,V] = eig(L_k(:,:,f));
      v = diag(V);
      vbar = max(v - lambdaL*mu, 0);
      Vbar = diag(vbar);
      L_k1(:,:,f) = U*Vbar*U.';
    end
    % (24): update Psi TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
    % TODO - verify with foti et al 2016 ref [7]
    Psi_k1 = zeros(p,p,F);
    for f = 1:F
      A = Psi_k + mu*LambdaPsi_k;
      Psi_k1(:,:,f) = pos((1-lambdaPsi*mu)./normsAdotij(A)) .* A(:,:,f);
    end


    % --- update duals (Ztilde = [Rtilde[.], PsiTilde[.], Ltilde[.]]) ---

    % (25-27): calculate Rbar, PsiBar, Lbar
    Rbar_k = R_k1 - mu*LambdaR_k;
    Psibar_k = Psi_k1 - mu*LambdaPsi_k;
    Lbar_k = L_k1 - mu*LambdaL_k;

    % (28): update Rtilde
    Rtilde_k1 = Rbar_k - (Rbar_k - Psibar_k + Lbar_k) / 3;
    % (29): update Psitilde
    Psitilde_k1 = Psibar_k - (Rbar_k + Psibar_k + Lbar_k) / 3;
    % (30): update Ltilde
    Ltilde_k1 = Lbar_k - (Rbar_k - Psibar_k + Lbar_k) / 3;


    % --- update splitting variable (Lambda) ---
    Z_k1 = [R_k1, Psi_k1, L_k1];
    Ztilde_k1    = [Rtilde_k1, Psitilde_k1, Ltilde_k1];
    Lambda_k1    = Lambda_k - (Z_k1 - Ztilde_k1) / mu;
    LambdaR_k1   = Lambda_k1(:,1:p,:);
    LambdaPsi_k1 = Lambda_k1(:,p+1:2*p,:);
    LambdaL_k1   = Lambda_k1(:,2*p+1:end,:);


    % --- check convergence ---
    i = i + 1;
    % update mu
    mu = mu - mu_dim_factor; % see foti et al. 2016 ref [15]
    % calculate primal/dual residuals
    r_k = sum(R_k - Psi_k + L_k,3);
    s_k = Ztilde_k1 - Ztilde_k;
    % (15) update epsilon
    %Z_k1_norms = sqrt(sum(sum(Z_k.^2,1),2)); % TODO Z_k or Z_k1?
    %Ztilde_k1_norms = sqrt(sum(sum(Z_k1.^2,1),2)); % TODO Ztilde_k or Ztilde_k1?
    %Lambda_k1_norms = sqrt(sum(sum(Lambda_k1.^2,1),2)); % TODO Lambda_k or Lambda_k1
    %epri = sqrt(p^2*F)*eabs + erel*sum(max(Z_k1_norms,Ztilde_k1_norms));
    %edual = sqrt(p^2*F)*eabs + erel*sum(Lambda_k1_norms);
    % check convergence
    terminate = i > maxiter;% || (norm(r_k) < epri && norm(s_k) < edual);

    
    % --- debug plot ---
    if debugPlot
      S = R_k + L_k;
      L = L_k;
      clf;
      subplot(211); imagesc(abs(S(:,:,floor(F/2)))); axis image;
      title(sprintf('Iteration %d',i));
      subplot(212); imagesc(abs(L(:,:,floor(F/2)))); axis image;
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
  info.niter = i;
  L = L_k;
  S = R_k + L_k;
  
  
end