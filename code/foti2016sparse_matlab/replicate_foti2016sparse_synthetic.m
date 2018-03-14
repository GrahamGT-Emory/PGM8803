% replicates experiments in section 6.1 of Foti et al. 2016
% figures 3 and 4

% construct data for VAR(1) synthetic data experiment in Foti et al.
% 2016, section 6.1

% --- params ---
p = 36; %69;
r = 5;
T = 128; %512;
snr = 2;

% --- construct data ---
% construct Astar
A = diag(0.2*ones(p,1));
for i = 1:p
  ind = randperm(p,2);
  A(i,ind) = round(rand(1,2))-1/2;
end
B = 2*randn(p,r);
for i = 1:r
  ind = randperm(p,round(0.2*p));
  B(ind,i) = 0;
end
C = zeros(r,p);
D = diag(randn(r,1));
Astar = [A B; C D];
Astar = Astar / max(eig(Astar));
% calculate x and u
sigma = norm(Astar,2) / snr; % todo - check this is spectral norm
XU = zeros(p+r,T);
XU(:,1) = randn(p+r,1);
for t = 2:T
  e = sigma*randn(p+r,1);
  XU(:,t) = Astar*XU(:,t-1) + e;
end
X = XU(1:p,:);
U = XU(p+1:end,:);
clearvars A B C D Astar e i ind sigma snr t XU


% --- solve with lvglasso ---
% C = 1/(T-1)*(X*X');
% params_lvglasso.alpha_sweep = 1; %logspace(-3,3,5);
% params_lvglasso.beta_sweep = 1; %logspace(-3,3,5);
% for ia = 1:length(params_lvglasso.alpha_sweep)
%   for ib = 1:length(params_lvglasso.beta_sweep)
%     fprintf('alpha %d/%d, beta %d/%d\n', ...
%       ia, length(params_lvglasso.alpha_sweep), ...
%       ib, length(params_lvglasso.beta_sweep));
%     alpha = params_lvglasso.alpha_sweep(ia);
%     beta = params_lvglasso.beta_sweep(ib);
%     [S,L,info] = lvglasso(C, alpha, beta);
%   end
% end


% --- solve with lvsglasso ---
Psi = calculateSpectralDensity(X,64);
[L,S] = lvsglasso_admm(Psi,1,1,true);


