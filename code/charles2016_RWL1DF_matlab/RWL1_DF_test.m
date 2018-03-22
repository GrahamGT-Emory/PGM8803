%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RWL1_DF_test 
%
% This code generates a sparse, dynamic test sequence with compressible
% innovations of various degrees. It then recovers the sequence from 
% compressible measurements (random Gaussian measurement matrices). This
% code compares recovery using time independent BPDN and RWL1 at each
% iteration, BPDN-DF (BPDN with a time dependent norm) and RWL1-DF.
% 
%
% 
%
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 06, 2012. 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initializations and Setup
clear
clc

% Set signal sizes and measurement numbers
x_dim = 500;       % Size of time-varying signal
s_num = 20;        % Sparsity of time-varying signal
num_iters = 100;   % Number of time steps to simulate and recover
num_trials = 1;   % Number of trials per test to run

% Set noise variables
obs_var = 0.001;   % Variance of the measurement noise
dyn_var = 0.01;    % Gaussian innovation value (if gaussian innovations)

% Initailize results vectors
x_store = zeros(x_dim, num_iters+1, num_trials);
x_est_all1 = zeros(x_dim, num_iters+1, num_trials);
x_spr_all1 = zeros(x_dim, num_iters+1, num_trials);
x_out_all = zeros(x_dim, num_iters+1, num_trials);
x_lsm1_all = zeros(x_dim, num_iters+1, num_trials);
x_rwl1_all = zeros(x_dim, num_iters+1, num_trials);
x_cs_all = zeros(x_dim, num_iters+1, num_trials);
x_kal_all = zeros(x_dim, num_iters+1, num_trials);

% Some useful parameters
tau = 1.1*obs_var;
tau2 = 1;

% poi_vec and y_dim_vec control the test being run. poi_vec controls the
% number of large random deviations from the model (innovations sparsity)
% and y_dim_vec controls the number of measurements. At least onr of these 
% must be a scalar, while the other could be a scalar or vector.

poi_vec = 3;
y_dim_vec = 70;
% poi_vec = 3;
% y_dim_vec = 10:5:200;

% Initialize vectors for storing the results

sweep_num = max(numel(y_dim_vec), numel(poi_vec));

Fin_Mat = zeros(sweep_num, 7);
Fin_MatV = zeros(sweep_num, 7);
Fin_MatMx = zeros(sweep_num, 7);
Fin_MatMn = zeros(sweep_num, 7);
Fin_Mat2 = zeros(sweep_num, 7);
Fin_Mat3 = zeros(sweep_num, 7);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run Recovery Algorithms
for mm = 1:sweep_num
    if numel(y_dim_vec) == 1
        p = poi_vec(mm);
        y_dim = y_dim_vec;
    elseif numel(poi_vec) == 1
        y_dim = y_dim_vec(y_d);
        p = poi_vec;
    else
        error('Code does not support sweeping both y_dim and poi_vec!')
    end

    for kk = 1:num_trials
        [x, y, G, F] = rand_sparse_seq(x_dim, y_dim, s_num, 1, num_iters, obs_var, 0*dyn_var, p);
        x_store(:, :, kk) = x;
        

        % Run BPDN-DF with sparse innovations and states
        [x_est1, ~] = BPDN_DF_L2L1(y, G, F, 0.5*tau, obs_var/dyn_var, 1000); 
        x_est_all1(:, :, kk) = x_est1;
        
        % Run BPDN-DF with sparse innovations only
        [x_spr1, ~] = BPDN_DF_L1(y, G, F, 0.5*tau, 1000);
        x_spr_all1(:, :, kk) = x_spr1;
        
        % Run BPDN-DF with sparse states only
        [x_out, ~] = BPDN_DF_L1L1(y, G, F, [0.5*obs_var, 0.001/(p+1)]);
        x_out(abs(x_out) <= 10^(-2)) = 0;

        % Run BPDN with no assumptions
        x_out_all(:, :, kk) = x_out;
        [x_cs, ~] = BPDN_multi(y, G, 0.5*tau, 1000);
        x_cs_all(:, :, kk) = x_cs;

        % Run RWL1_DF with dynamics
        [x_lsm1, lambda_est1, ~] = RWL1_DF(y, G, F, 1*(dyn_var), obs_var, 20, tau, 0.01);
        x_lsm1_all(:, :, kk) = x_lsm1;
        
        % Run RWL1 (RWL1_DF without dynamics)
        [x_rwl1, lambda_est2, ~] = RWL1_DF(y, G, 0*F, 100.*(dyn_var), obs_var, 20, tau, 0.01);
        x_rwl1_all(:, :, kk) = x_rwl1;
        
        % Run the Kalman filter
        [x_kalman, ~] = Basic_KF(y, G, F, dyn_var, obs_var);
        x_kal_all(:, :, kk) = x_kalman;
        
    end
    
    % Parse/save the ouputs
    x_norms = sum(x_store.^2, 1);
    temp = [mean(sum((x_store - x_est_all1).^2, 1)./x_norms, 3).',... 
        mean(sum((x_store - x_spr_all1).^2, 1)./x_norms, 3).',... 
        mean(sum((x_store - x_out_all).^2, 1)./x_norms, 3).', ...
        mean(sum((x_store - x_lsm1_all).^2, 1)./x_norms, 3).', ...
        mean(sum((x_store - x_cs_all).^2, 1)./x_norms, 3).',...
        mean(sum((x_store - x_rwl1_all).^2, 1)./x_norms, 3).', ...
        mean(sum((x_store - x_kal_all).^2, 1)./x_norms, 3).'];
    Fin_Mat(mm, :) = temp(end-1, :);
    Fin_Mat2(mm, :) = mean(temp(end - 20: end-1, :), 1);
    
    for ll = 1:size(x_est_all1, 3)
        temp2 = [(sum((x_store(:, :, ll) - x_est_all1(:, :, ll)).^2, 1)./x_norms(:, :, ll)).',... 
            (sum((x_store(:, :, ll) - x_spr_all1(:, :, ll)).^2, 1)./x_norms(:, :, ll)).',... 
            (sum((x_store(:, :, ll) - x_out_all(:, :, ll)).^2, 1)./x_norms(:, :, ll)).', ...
            (sum((x_store(:, :, ll) - x_lsm1_all(:, :, ll)).^2, 1)./x_norms(:, :, ll)).', ...
            (sum((x_store(:, :, ll) - x_cs_all(:, :, ll)).^2, 1)./x_norms(:, :, ll)).',...
            (sum((x_store(:, :, ll) - x_rwl1_all(:, :, ll)).^2, 1)./x_norms(:, :, ll)).', ...
            (sum((x_store(:, :, ll) - x_kal_all(:, :, ll)).^2, 1)./x_norms(:, :, ll)).'];
        end_vals = mean(temp2(end-20: end-1, :), 1);
   
        for kk = 1:size(temp, 2)
            Fin_Mat3(mm, kk) = find((temp2(:, kk) > 1.05*end_vals(kk)) == 0, 1);
        end
    end

    temp = [var(sum((x_store - x_est_all1).^2, 1)./x_norms, [], 3).',... 
        var(sum((x_store - x_spr_all1).^2, 1)./x_norms, [], 3).',... 
        var(sum((x_store - x_out_all).^2, 1)./x_norms, [], 3).', ...
        var(sum((x_store - x_lsm1_all).^2, 1)./x_norms, [], 3).', ...
        var(sum((x_store - x_cs_all).^2, 1)./x_norms, [], 3).',...
        var(sum((x_store - x_rwl1_all).^2, 1)./x_norms, [], 3).', ...
        var(sum((x_store - x_kal_all).^2, 1)./x_norms, [], 3).'];
    Fin_MatV(mm, :) = temp(end-1, :);
    
    temp = [max(sum((x_store - x_est_all1).^2, 1)./x_norms, [], 3).',... 
        max(sum((x_store - x_spr_all1).^2, 1)./x_norms, [], 3).',... 
        max(sum((x_store - x_out_all).^2, 1)./x_norms, [], 3).', ...
        max(sum((x_store - x_lsm1_all).^2, 1)./x_norms, [], 3).', ...
        max(sum((x_store - x_cs_all).^2, 1)./x_norms, [], 3).',...
        max(sum((x_store - x_rwl1_all).^2, 1)./x_norms, [], 3).', ...
        max(sum((x_store - x_kal_all).^2, 1)./x_norms, [], 3).'];
    Fin_MatMx(mm, :) = temp(end-1, :);
    
    temp = [min(sum((x_store - x_est_all1).^2, 1)./x_norms, [], 3).',... 
        min(sum((x_store - x_spr_all1).^2, 1)./x_norms, [], 3).',... 
        min(sum((x_store - x_out_all).^2, 1)./x_norms, [], 3).', ...
        min(sum((x_store - x_lsm1_all).^2, 1)./x_norms, [], 3).', ...
        min(sum((x_store - x_cs_all).^2, 1)./x_norms, [], 3).',...
        min(sum((x_store - x_rwl1_all).^2, 1)./x_norms, [], 3).', ...
        min(sum((x_store - x_kal_all).^2, 1)./x_norms, [], 3).'];
    Fin_MatMn(mm, :) = temp(end-1, :);
    
    
    fprintf('Iteration %d of %d done.', mm, sweep_num)
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting Code

if (numel(y_dim_vec) == 1)&&(numel(poi_vec) == 1)
    %% Plotting through time
    BPDN_MRMSE = mean(sum((x_store - x_cs_all).^2, 1)./x_norms, 3).';
    RWL1_MRMSE = mean(sum((x_store - x_rwl1_all).^2, 1)./x_norms, 3).';
    BPDNDF_MRMSE = mean(sum((x_store - x_out_all).^2, 1)./x_norms, 3).';
    RWL1DF_MRMSE = mean(sum((x_store - x_lsm1_all).^2, 1)./x_norms, 3).';

    % figure(1)
    % close 1
    figure(1), hold off
    figure(1)
    plot(0, BPDN_MRMSE(1), '.:b', 'LineWidth', 3)
    figure(1), hold on;
    plot(0, RWL1_MRMSE(1), 'v-g', 'LineWidth', 3)
    plot(0,BPDNDF_MRMSE(1), 'd-.r', 'LineWidth', 3)
    plot(0, RWL1DF_MRMSE(1), 's--c', 'LineWidth', 3) 

    plot(0:num_iters, BPDN_MRMSE, ':b', 'LineWidth', 3)
    plot(0:num_iters, RWL1_MRMSE, '-g', 'LineWidth', 3)
    plot(0:num_iters,BPDNDF_MRMSE, '-.r', 'LineWidth', 3)
    plot(0:num_iters, RWL1DF_MRMSE, '--c', 'LineWidth', 3) 

    plot(0:3:num_iters, BPDN_MRMSE(1:3:end), '.k', 'LineWidth', 3)
    plot(0:3:num_iters, RWL1_MRMSE(1:3:end), 'vg', 'LineWidth', 3)
    plot(0:3:num_iters,BPDNDF_MRMSE(1:3:end), 'dr', 'LineWidth', 3)
    plot(0:3:num_iters, RWL1DF_MRMSE(1:3:end), 'sc', 'LineWidth', 3) 


    legend('BPDN', 'RWL1', 'BPDN-DF', ...
        'RWL1-DF')
    xlabel('Time Index', 'FontSize', 30)
    ylabel('Mean rMSE', 'FontSize', 30)
    set(gca, 'FontSize', 23, 'Ylim', [0,0.25], 'Xlim', ...
        [0,num_iters])

elseif (numel(y_dim_vec) ~= 1)&&(numel(poi_vec) == 1)
    %% Plot sweeping through the innovations sparsity
    LL_pt = 10;
    UL_pt = 35;

    % figure(1)
    % close 1
    figure(1), hold off;
    figure(1)
    plot(y_dim_vec(LL_pt:UL_pt), Fin_Mat2(LL_pt:UL_pt, 5), '.:b', 'LineWidth', 3)
    figure(1), hold on;
    plot(y_dim_vec(LL_pt:UL_pt), Fin_Mat2(LL_pt:UL_pt, 6), 'v-g', 'LineWidth', 3)
    plot(y_dim_vec(LL_pt:UL_pt), Fin_Mat2(LL_pt:UL_pt, 3), 'd-.r', 'LineWidth', 3)
    plot(y_dim_vec(LL_pt:UL_pt), Fin_Mat2(LL_pt:UL_pt, 4), 's--c', 'LineWidth', 3) 

    legend('BPDN', 'RWL1', 'BPDN-DF', ...
        'RWL1-DF')
    xlabel('Number of Measurements', 'FontSize', 30)
    ylabel('Steady State rMSE', 'FontSize', 30)
    set(gca, 'FontSize', 23, 'Ylim', [0,0.2], 'Xlim', ...
        [y_dim_vec(LL_pt), y_dim_vec(UL_pt)])
elseif (numel(y_dim_vec) == 1)&&(numel(poi_vec) ~= 1)
    %% Plot sweeping through the number of measurements
    LL_pt = 1;
    UL_pt = 5;

    figure(1), hold off
    figure(1)
    plot(poi_vec(LL_pt:UL_pt), Fin_Mat2(LL_pt:UL_pt, 5), '.:b', 'LineWidth', 3)
    figure(1), hold on;
    plot(poi_vec(LL_pt:UL_pt), Fin_Mat2(LL_pt:UL_pt, 6), 'v-g', 'LineWidth', 3)
    plot(poi_vec(LL_pt:UL_pt), Fin_Mat2(LL_pt:UL_pt, 3), 'd-.r', 'LineWidth', 3)
    plot(poi_vec(LL_pt:UL_pt), Fin_Mat2(LL_pt:UL_pt, 4), 's--c', 'LineWidth', 3) 

    legend('BPDN', 'RWL1', 'BPDN-DF', ...
        'RWL1-DF')
    xlabel('Number of Measurements', 'FontSize', 30)
    ylabel('Steady State rMSE', 'FontSize', 30)
    set(gca, 'FontSize', 23, 'Ylim', [0,0.25], 'Xlim', ...
        [1, 5])
    
else
    error('Code does not support sweeping both y_dim and poi_vec!')
end




