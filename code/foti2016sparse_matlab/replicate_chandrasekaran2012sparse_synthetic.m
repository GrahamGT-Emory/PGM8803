
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

p = 198;
h = 2;
S = zeros(p+h);
r = 0.2;
% p random locations on a 1x1 square
locs = rand(198, 2);
% Add partial correlation edges 
for ii = 1:p
    for jj = 1:p
        if ii < jj
            add_edge = rand() < (2 * normpdf(norm(locs(ii,:) - locs(jj,:),2) * sqrt(p)));
            S(ii, jj) = r * add_edge;
            S(jj,ii) = r * add_edge;
        elseif ii == jj
            S(ii, jj) = 1;
        end
    end
end
% Latent variables connected to random subset of observed variables
subset_percent = 1;
for ii = (p+1):(p+h)
    ind = randperm(p,round(subset_percent*p));
    vals = 0.12*rand(1,round(subset_percent*p));
    S(ind,ii) = vals;
    S(ii, ind) = vals;
    for jj = (p+1):(p+h)
        if ii == jj
            S(ii, jj) = 1;
        else
            S(ii, jj) = r;
        end
        
    end
end

C = inv(S);
n = 10*p;
% XU is [p+h x n] synthetic data
XU = (randn(n, p+h) * chol(C))'; 
X = XU(1:p,:);

% --- solve with lvglasso ---
C_sample = 1/(n-1)*(X*X');
params_lvglasso.alpha_sweep = 1; %logspace(-3,3,5);
params_lvglasso.beta_sweep = 1; %logspace(-3,3,5);
for ia = 1:length(params_lvglasso.alpha_sweep)
  for ib = 1:length(params_lvglasso.beta_sweep)
    fprintf('alpha %d/%d, beta %d/%d\n', ...
      ia, length(params_lvglasso.alpha_sweep), ...
      ib, length(params_lvglasso.beta_sweep));
    alpha = params_lvglasso.alpha_sweep(ia);
    beta = params_lvglasso.beta_sweep(ib);
    [S_predicted,L,info] = lvglasso(C_sample, alpha, beta);
  end
end
figure;
plot(graph(S(1:p,1:p)),'XData',locs(:,1),'YData',locs(:,2));
title('True Graph')
figure;
plot(graph(S_predicted(1:p,1:p)),'XData',locs(:,1),'YData',locs(:,2));
title('Predicted')