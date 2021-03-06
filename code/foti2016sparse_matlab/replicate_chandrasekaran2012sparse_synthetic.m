
% replicates experiments in section 6.1 of Chadrasekaran et al. 2012
% figure 1

% %% construct data for synthetic data experiment in Chadrasekaran et al.
% % 2012, section 6.1 - 36-cycle, 2 latent variables
% p = 36;
% h = 2;
% S = zeros(p+h);
% r = 0.25;
% % Add partial correlation edges 
% for ii = 0:(p-1)
%     S(ii, mod(ii+1,p)) = r;
%     S(mod(ii+1,p), ii) = r;
%     S(ii, mod(ii-1,p)) = r;
%     S(mod(ii-1,p), ii) = r;
% end
% % Latent variables connected to random subset of observed variables
% subset_percent = 0.8;
% for ii = 0:(h-1)
%     ind = randperm(p,round(subset_percent*p));
%     S(ind,ii) = rand(1,round(subset_percent*p));
%     S(ii, ind) = rand(1,round(subset_percent*p));
% end
% C = inv(S);
% n = 100*p;
% % XU is [p+h x n] synthetic data
% XU = (randn(n, p+h) * chol(C))'; 
% 
% SampleC = XU(1:p,:);

%% construct data for synthetic data experiment in Chadrasekaran et al.
% 2012 - Ming Yuan Georgia Tech, section 6.1 - 36-cycle, 2 latent variables
clc; clear;
addpath('../ma2013admlvglasso_matlab/');
p = 50;
h = 5;
S = eye(p+h);
r = 0.2;
% p random locations on a 1x1 square
locs = rand(p, 2);
% Add partial correlation edges 
for ii = 1:p
    for jj = 1:p
        if ii < jj
            add_edge = rand() < (2 * normpdf(norm(locs(ii,:) - locs(jj,:),2) * sqrt(p)));
            S(ii, jj) = r * add_edge;
        end
    end
end
% Remove edges when nodes have more than four edges connected to it
max_edges = 4;
for ii = 1:p
    [node_edges] = find(S(ii,:) > 0);
    if length(node_edges) > max_edges
        idx_delete = randperm(length(node_edges));
        S(ii, node_edges(idx_delete(max_edges+1:end))) = 0;
        S(node_edges(idx_delete(max_edges+1:end)),ii) = 0;
    end 
    S(ii, ii) = 1;
end
S = (S + S') - (eye(p+h).*diag(S)); 

% Latent variables connected to random subset of observed variables
subset_percent = 1;
for ii = (p+1):(p+h)
    ind = 1:p;
    ind = randperm(p,round(subset_percent*p));
    vals = 0.11*rand(1,round(subset_percent*p));
    vals = 0.11*rand(1,round(subset_percent*p));
    S(ind,ii) = vals;
    S(ii, ind) = vals;
    for jj = (p+1):(p+h)
        if ii == jj
            S(ii, jj) = 1;
        else
            % Latent Variables connection weights
            S(ii, jj) = 0;
        end  
    end
end

C = inv(S);
n = 100*round(0.6*(p.^2)); % from meinshausen 2006, pg 1449
% X is [p x n] synthetic data
X = (randn(n, p) * chol(C(1:p,1:p)))';
XU = (randn(n, p+h) * chol(C))';
threshold = 1e-9;

%% remove this cell
specden = calculateSpectralDensity(XU,16,ones(1,16));
subplot(311); imagesc(abs(S)); title('actual');
subplot(312); imagesc(sum(abs(specden),3)); title('mean,3 of specden');
subplot(313); imagesc(all(abs(specden)>10^-5.2,3)); title('thresholded');



%% Optimization

C_sample = (1/(n-1))*(X*X');
params_glasso.alpha_sweep = logspace(-1,2.4,10);
params_lvglasso.alpha_sweep = logspace(-1,2.4,4);
params_lvglasso.beta_sweep = logspace(-1,3,4);
lv_history = [];
glasso_history = [];

opts.continuation = 1; opts.mu = n; opts.num_continuation = 10; opts.eta = 1/4; opts.muf = 1e-6;
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
%     [S_lv,L_lvglasso,info_lvglasso] = lvglasso(C_sample, alpha, beta);
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

for ia = 1:length(params_glasso.alpha_sweep)
    alpha = params_glasso.alpha_sweep(ia);
    % --- solve with glasso ---
%     [S_glasso,info_glasso] = glasso(C_sample, alpha);
    out_GL = ADMM_GLASSO(C_sample,alpha,opts);
    S_glasso = out_GL.R;  
    info_glasso = struct();  info_glasso.obj = out_GL.obj;
    [f1_score] = evaluateGraph(abs(S_glasso) > threshold, targets);
    info_glasso.obj = f1_score;
    
    % Create struct
    meta_data = struct();
    meta_data.alpha = alpha;
    meta_data.f1 = f1_score;
    meta_data.S = S_glasso;
    meta_data.info = info_glasso;
    meta_data.obj = info_glasso.obj;
    glasso_history = [glasso_history meta_data]; %#ok<*AGROW>
end

%% Plot results

close all;
subplot 221
G = graph(abs(S(1:p,1:p)) > threshold,'OmitSelfLoops');
plot(G,'XData',locs(:,1),'YData',locs(:,2));
title('True Graph')

subplot 222
bic_lv = log(n)*length(lv_history) - 2*log([lv_history.obj]);
[~, loc] = max(bic_lv);
G = graph(abs(lv_history(loc).S) > threshold,'OmitSelfLoops');
plot(G,'XData',locs(:,1),'YData',locs(:,2));
title('LV-Glasso')

subplot 223
bic_glasso = log(n)*length(glasso_history) - 2*log([glasso_history.obj]);
[~, loc] = min(bic_glasso);
G = graph(abs(glasso_history(loc).S) > threshold,'OmitSelfLoops');
plot(G,'XData',locs(:,1),'YData',locs(:,2));
title('Glasso')



