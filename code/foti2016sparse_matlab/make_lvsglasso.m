function make_lvsglasso(F)

  fh = fopen(sprintf('lvsglasso_%d.m',F), 'w');
  
  fprintf(fh,'%% LVSGLASSO sparse + low-rank time series graphical lasso (foti et al 2016)\n');
  fprintf(fh,'%% INPUTS\n');
  fprintf(fh,'%%  S      : spectral density matrix [p x p x F]\n');
  fprintf(fh,'%%  lPsi   : trade-off parameter for l1 norm\n');
  fprintf(fh,'%%  lL     : trade-off parameter for trace(L)\n');
  fprintf(fh,'%% OUTPUTS\n');
  fprintf(fh,'%%  S      : predicted observed precision matrix\n');
  fprintf(fh,'%%  L      : predicted hidden precision matrix\n');
  fprintf(fh,'%%  info   : CVX debug information\n');
  fprintf(fh,'\n');
  fprintf(fh,'function [S,L,info] = lvsglasso(S, lambdaPsi, lambdaL)\n');
  fprintf(fh,'  \n');
  fprintf(fh,'  [p,~,F] = size(S);\n');
  fprintf(fh,'  triu_mask = repmat(logical(ones(p)-tril(ones(p))),F);\n');
  fprintf(fh,'  \n');
  fprintf(fh,'  cvx_begin sdp\n');
  fprintf(fh,'    variable Psi(p,p,F)\n');
  fprintf(fh,'    variable L(p,p,F)\n');
  fprintf(fh,'    minimize ( ...\n');
  for i = 1:F
    fprintf(fh,'      - log_det(Psi(:,:,%d) - L(:,:,%d)) + trace(S(:,:,%d)*(Psi(:,:,%d)-L(:,:,%d))) ...\n',i,i,i,i,i);
  end
  fprintf(fh,'      + lambdaPsi * norm(Psi(triu_mask),1) ...\n');
  for i = 1:F-1
    fprintf(fh,'      + lambdaL*trace(L(:,:,%d)) ...\n', i);
  end
  fprintf(fh,'      + lambdaL*trace(L(:,:,%d)) )\n', F);
  fprintf(fh,'    subject to\n');
  fprintf(fh,'      for i = 1:F\n');
  fprintf(fh,'        S(:,:,i) - L(:,:,i) > 0\n');
  fprintf(fh,'        L(:,:,i) >= 0\n');
  fprintf(fh,'      end\n');
  fprintf(fh,'   cvx_end\n');
  fprintf(fh,'\n');
  fprintf(fh,'end');
  
  fclose(fh);


end