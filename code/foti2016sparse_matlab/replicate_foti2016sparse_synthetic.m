% replicates experiments in section 6.1 of Foti et al. 2016
% figures 3 and 4

% construct data for VAR(1) synthetic data experiment in Foti et al.
% 2016, section 6.1

% Add ADMM implementation of LV-glasso
addpath('../ma2013admlvglasso_matlab/');

% --- params ---
p = 69;
r = 5;
T = 10000;
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
clearvars A B C D Astar e i ind sigma snr t 


% To avoid numerical errors, sets low values to 0 for plotting graphs
C_actual = (1/(T-1))*(XU*XU');
S = inv(C_actual);
threshold = quantile(real(S(:)),[0.95]);
num_samples = 512;
C_sample = (1/(num_samples-1))*(X(:,1:num_samples)*X(:,1:num_samples)');
params_lvglasso.alpha_sweep = logspace(-1,2.4,5);
params_lvglasso.beta_sweep = logspace(-1,3,5);
lv_history = [];


opts.continuation = 1; opts.mu = num_samples; opts.num_continuation = 10; opts.eta = 1/4; opts.muf = 1e-6;
opts.maxiter = 500; opts.stoptol = 1e-5; opts.over_relax_par = 1.6; 
targets = abs(S(1:p, 1:p)) > threshold;

for ia = 1:length(params_lvglasso.alpha_sweep)
  for ib = 1:length(params_lvglasso.beta_sweep)
    alpha = params_lvglasso.alpha_sweep(ia);
    beta = params_lvglasso.beta_sweep(ib);
    % --- solve with lvglasso ---
    out_B = ADMM_B(C_sample,alpha,beta,opts);
    S_lv = out_B.S;  L_lvglasso = out_B.L; 
    info_lvglasso = struct();  info_lvglasso.obj = out_B.obj;
    [f1_score] = evaluateGraph(abs(S_lv) > threshold, targets);
    
    % Create struct
    meta_data = struct();
    meta_data.alpha = alpha;
    meta_data.beta = beta;
    meta_data.f1 = f1_score;
    meta_data.S = S_lv;
    meta_data.L = L_lvglasso;
    meta_data.info = info_lvglasso;
    meta_data.obj = info_lvglasso.obj;
    lv_history = [lv_history meta_data]; %#ok<*AGROW>
  end
end


% % --- solve with lvsglasso ---
% Psi = calculateSpectralDensity(X,64);
% [L,S] = lvsglasso_admm(Psi,1,1,true);


%% Plot results

close all;
subplot 221
G = graph(abs(S(1:p,1:p)) > threshold,'OmitSelfLoops');
plot(G,'','NodeLabel',{});
title('True Graph')

subplot 222
bic_lv = log(num_samples)*length(lv_history) - 2*log([lv_history.obj]);
[~, loc] = max([lv_history.f1]);
G = graph(abs(lv_history(loc).S) > threshold,'OmitSelfLoops');
plot(G,'','NodeLabel',{});
title('LV-Glasso')






