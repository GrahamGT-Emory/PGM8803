% replicates experiments in section 6.1 of Foti et al. 2016
% figures 3 and 4

% construct data for VAR(1) synthetic data experiment in Foti et al.
% 2016, section 6.1

%% --- construct data ---
% Add ADMM implementation of LV-glasso
%addpath('../ma2013admlvglasso_matlab/');
%clear; clc; close all;

% --- params ---
p = 50;
r = 5;
T = 1000;
snr = 200;


% --- construct Astar for VAR(1) process ---
A = diag(0.2*ones(p,1));
for i = 1:p
  ind = randperm(p,2);
  while any(ind == i)
      ind = randperm(p,2);
  end
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


% --- generate data from VAR(1) process ---
sigma = norm(Astar,2) / snr; % todo - check this is spectral norm
XU = zeros(p+r,T);
XU(:,1) = randn(p+r,1);
for t = 2:T
  e = sigma*randn(p+r,1);
  XU(:,t) = Astar*XU(:,t-1) + e;
end
X = XU(1:p,:);
U = XU(p+1:end,:);


% --- calculate estimated spectral density matrix ---
fk = calculateSpectralDensity(X.');
invfk = zeros(size(fk));
for i = 1:size(fk,3)
  invfk(:,:,i) = inv(fk(:,:,i));
end
fk = permute(fk,[3 1 2]);
invfk = permute(fk,[3 1 2]);
save('lvsglasso_inputs','fk','invfk');


% %% visualize estimated spectral density matrix
% fkinv = zeros(size(fk));
% for i = 1:size(fk,3)
%   fkinv(:,:,i) = inv(fk(:,:,i));
% end
% 
% 
% %% --- lvglasso run ---
% params_lvglasso.alpha_sweep = 0.1;%logspace(-1,0,2);
% params_lvglasso.beta_sweep = 1000;%logspace(-1,3,2);
% lv_history = [];
% 
% 
% opts.continuation = 1; opts.mu = num_samples; opts.num_continuation = 10; opts.eta = 1/4; opts.muf = 1e-6;
% opts.maxiter = 500; opts.stoptol = 1e-5; opts.over_relax_par = 1.6; 
% 
% for ia = 1:length(params_lvglasso.alpha_sweep)
%   for ib = 1:length(params_lvglasso.beta_sweep)
%     alpha = params_lvglasso.alpha_sweep(ia);
%     beta = params_lvglasso.beta_sweep(ib);
%     % --- solve with lvglasso ---
%     out_LV = ADMM_B(C_sample,alpha,beta,opts);
%     S_lv = out_LV.S;  L_lvglasso = out_LV.L; 
%     info_lvglasso = struct();  info_lvglasso.obj = out_LV.obj;
%     [f1_score_lv] = evaluateGraph(abs(S_lv) > threshold, targets);
%     
%     % Create struct
%     meta_data = struct();
%     meta_data.alpha = alpha;
%     meta_data.beta = beta;
%     meta_data.f1 = f1_score_lv;
%     meta_data.S = S_lv;
%     meta_data.L = L_lvglasso;
%     meta_data.info = info_lvglasso;
%     meta_data.obj = info_lvglasso.obj;
%     lv_history = [lv_history meta_data]; %#ok<*AGROW>
%   end
% end
% 
% 
% %% --- lvsglasso run ---
% F = 60;
% windowSize = 10;
% spectralDensity = 'compute';
% if strcmpi(spectralDensity,'compute')
%   S_hat = calculateSpectralDensity(XU(:,1:num_samples),F);
% elseif strcmpi(spectralDensity,'matlab')
%   [S_hat] = cpsd(XU(1:end,1:num_samples)',[XU(1:end,1:num_samples)' zeros(1, num_samples)'],windowSize ,[],F);
%   S_hat(:, :, end) = [];
%   S_hat = permute(S_hat,[2 3 1]);
% end
% S_inv = zeros(size(S_hat));
% for f = 1:size(S_hat,3)
%   S_inv(:,:,f) = inv(S_hat(:,:,f));
% end
% threshold_spectral = quantile(abs(real(S_inv(:))),[0.90]);
% [f1_score] = evaluateGraph(~all(abs(real(S_inv(1:p,1:p,:))) < threshold_spectral,3), targets);
% 
% opts_lvsglasso.continuation = 0.1; opts_lvsglasso.mu = num_samples; 
% opts_lvsglasso.num_continuation = 10; opts_lvsglasso.eta = 1/4; opts_lvsglasso.muf = 1e-6;
% opts_lvsglasso.maxiter = 100; opts_lvsglasso.stoptol = 1e-5; opts_lvsglasso.over_relax_par = 1.6; 
% 
% % opts_lvsglasso.maxiter = 100; opts_lvsglasso.mu = num_samples; opts_lvsglasso.mu_dim_factor = 1/4;
% % opts_lvsglasso.debugPlot = false; opts_lvsglasso.over_relax_par = 1.6; 
% params_lvsglasso.alpha_sweep = 10^1.2;%logspace(0,2.4,2);
% params_lvsglasso.beta_sweep = 10;%logspace(-1,3,2);
% opts.debugPlot = false;
% lvs_history = [];
% 
% for ia = 1:length(params_lvsglasso.alpha_sweep)
%   for ib = 1:length(params_lvsglasso.beta_sweep)
%     alpha = params_lvsglasso.alpha_sweep(ia);
%     beta = params_lvsglasso.beta_sweep(ib);
%     % --- solve with lvsglasso ---
%     out_LVS = lvsglasso_admm_2(S_hat,alpha,beta,opts_lvsglasso);
%     S_lvs = out_LVS.S;  L_lvs = out_LVS.L; 
%     info_lvsglasso = struct();  info_lvsglasso.obj = out_LVS.obj;
%     %[f1_score] = evaluateGraph(~all(abs(real(S_lvs)) < threshold,3), targets);
%     
%     % Create struct
%     meta_data = struct();
%     meta_data.alpha = alpha;
%     meta_data.beta = beta;
%     meta_data.f1 = f1_score;
%     meta_data.S = S_lvs;
%     meta_data.L = L_lvs;
%     meta_data.info = info_lvsglasso;
%     meta_data.obj = info_lvsglasso.obj;
%     lvs_history = [lvs_history meta_data]; %#ok<*AGROW>
%   end
% end
% 
% 
% 
% 
% %% Plot results
% 
% clf;
% subplot 221
% S_true = 1/2*(S(1:p,1:p)+S(1:p,1:p)');
% G = graph(abs(S_sym(1:p,1:p)) > threshold,'OmitSelfLoops');
% plot(G,'','NodeLabel',{});
% title('True Graph');
% 
% subplot 222
% bic_lv = log(num_samples)*length(lv_history) - 2*log([lv_history.obj]);
% [bestLVVal, loc] = max([lv_history.f1]);
% S_best = lv_history(loc).S;
% G = graph(abs(S_best) > 0.01*threshold,'OmitSelfLoops');
% plot(G,'','NodeLabel',{});
% title('LV-Glasso');
% 
% subplot 223
% bic_lvs = log(num_samples)*length(lvs_history) - 2*log([lvs_history.obj]);
% [~, loc] = max([lvs_history.f1]);
% G = graph(~all(abs(real(S_lvs)) < 0.01*threshold,3),'OmitSelfLoops');
% plot(G,'','NodeLabel',{});
% title('LVS-Glasso');





