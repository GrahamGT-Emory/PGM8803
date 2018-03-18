
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

p = 15;
h = 2;
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
%             S(jj,ii) = r * add_edge;
        end
    end
end
% Remove edges when nodes have more than four edges connected to it
for ii = 1:p
    [node_edges] = find(S(ii,:) > 0);
    if length(node_edges) > 5
        idx_delete = randperm(length(node_edges));
        S(ii, node_edges(idx_delete(6:end))) = 0;
        S(node_edges(idx_delete(6:end)),ii) = 0;
    end 
    S(ii, ii) = 1;
end
S = (S + S') - (eye(p+h).*diag(S)); 

% Latent variables connected to random subset of observed variables
subset_percent = 1;
for ii = (p+1):(p+h)
    ind = randperm(p,round(subset_percent*p));
    vals = 0.3*rand(1,round(subset_percent*p));
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

S_obs_marginal = S(1:p,1:p) - (S(1:p,p+1:end)*inv(S(p+1:end,p+1:end))*S(p+1:end,1:p));
C = inv(S_obs_marginal);
% unity variances`

n = 50*(p); % from meinshausen 2006, pg 1449
% X is [p x n] synthetic data
X = (randn(n, p) * chol(C))'; 
threshold = 1e-9;
close all;


subplot 221
G = graph(S(1:p,1:p));
plot(G,'XData',locs(:,1),'YData',locs(:,2),'LineWidth', G.Edges.Weight);
title('True Graph')

%% Optimization
best_f1_lv = 0;
best_f1_glasso = 0;
C_sample = 1/(n-1)*(X*X');
params_lvglasso.alpha_sweep = 0.8;%logspace(-1,0,4);
params_lvglasso.beta_sweep = 0.3;%logspace(0,1,10);
for ia = 1:length(params_lvglasso.alpha_sweep)
    
  targets = abs(S(1:p, 1:p)) > threshold;
  for ib = 1:length(params_lvglasso.beta_sweep)
    fprintf('alpha %d/%d, beta %d/%d\n', ...
      ia, length(params_lvglasso.alpha_sweep), ...
      ib, length(params_lvglasso.beta_sweep));
    alpha = params_lvglasso.alpha_sweep(ia);
    beta = params_lvglasso.beta_sweep(ib);
    % --- solve with lvglasso ---
    [S_lv,L_lvglasso,info_lvglasso] = lvglasso(C_sample, alpha, beta);
    
    [fl_score] = evaluateGraph(abs(S_lv) > threshold, targets);
    if fl_score > best_f1_lv
        best_S_lv = S_lv;
        best_f1_lv = fl_score;
        best_param_lv = [ia, ib];
        subplot 222
        G = graph(abs(best_S_lv) > threshold);
        plot(G,'XData',locs(:,1),'YData',locs(:,2));
        title('LV-Glasso')
        pause(0.1)
    end
    % Only depends on first loop parameters
    if ib == 1
        % --- solve with glasso ---
        [S_glasso, info_glass] = glasso(C_sample, alpha);
        
        [fl_score] = evaluateGraph(abs(S_glasso) > threshold, targets);
        if fl_score > best_f1_glasso
            best_S_glasso = S_glasso;
            best_f1_glasso = fl_score;
            best_param_glasso = [ia];
            subplot 223
            G = graph(abs(best_S_glasso) > threshold);
            plot(G,'XData',locs(:,1),'YData',locs(:,2));
            title('Glasso')
            pause(0.1)
        end
    end
  end
end




