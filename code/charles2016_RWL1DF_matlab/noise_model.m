function noise_state = noise_model(pure_state, d_var, poi_var)

% noise_state = noise_model(pure_state, d_var, poi_var)
% 
% Applies the noise model to the pure state. Essentially the noise model 
% adds 0-mean, d_var variance Gaussian noise on the support and randomly
% replaces some number of the support with completely new support. The 
% number of replaced support is a Poisson random variable with mean poi_var
% 
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 21, 2012. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Apply noise model

if sum(pure_state == 0) ~= 0
% first add in random gaussian noise over the support
state_supp = pure_state ~= 0;
noise_state = pure_state + state_supp.*(sqrt(d_var)*randn(size(pure_state)));

% randomly change the support a bit
if poi_var > 0
    supp_change_num = rand_posn(1, 1, poi_var);
    if supp_change_num > sum(state_supp)
        supp_change_num = sum(state_supp);
    end

    supp_now = find(noise_state ~= 0);
    supp_now = supp_now(randperm(length(supp_now)));
    noise_state(supp_now(1:supp_change_num)) = 0;

    supp_now = supp_now(supp_change_num+1:end);
    while sum(noise_state ~= 0) < sum(state_supp)
        rec_add = ceil(rand(1)*length(noise_state));
        if sum((rec_add - supp_now) == 0) == 0
            supp_now = [supp_now; rec_add];
            noise_state(rec_add) = randn(1);
        end
    end
end

else
    temp = randperm(length(pure_state));
    noise_temp = zeros(size(pure_state));
    p = rand_posn(1, 1, poi_var);
    noise_temp(temp(1:p)) = randn(1, p);
    
    pure_state(temp(1:p)) = 0;
    noise_state = pure_state + noise_temp + sqrt(d_var)*randn(size(pure_state));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
