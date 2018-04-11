% Dynamic sparse + low-rank time series graphical lasso
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

% Our goal is to minimize the following objective function:
% min <R,SigmaO> - logdet(R) + alpha ||S||_1  + beta
% s.t. R positive definite and R - S + L = 0