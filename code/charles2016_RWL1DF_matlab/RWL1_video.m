function [varargout] = RWL1_video(varargin)

% [vid_coef_rwcs, vid_recon_rwcs, vid_rMSE_rwcs, vid_PSNR_rwcs] = ...
%           RWL1_video(MEAS_SIG, MEAS_SEL, lambda_val, TOL, DWTfunc, TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% param_vals: 1x3 vector of parameter values
% TOL:        Scalar value for the tolerance in the TFOCS solver
% DWTfunc:    Wavelet transform (sparsity basis)
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate rMSE and PSNR)
% 
%    The outputs are:
% 
% vid_coef_rwcs:  Nx1xT array of inferred sparse coefficients
% vid_recon_rwcs: Sqrt(N)xSqrt(N)xT array of the recovered video sequence
% vid_rMSE_rwcs:  Tx1 array of rMSE values for the recovered video
% vid_PSNR_rwcs:  Tx1 array of PSNR values for the recovered video
% 
% 
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated August 20, 2012. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse Inputs
MEAS_SIG = varargin{1};
MEAS_FUN = varargin{2};
param_vals = varargin{3}; 
TOL = varargin{4};
DWTfunc = varargin{5};

if nargin > 5
    rMSE_calc_opt = 1;
    TRUE_VID = varargin{6};
else
    rMSE_calc_opt = 0;
end

DWT_apply = DWTfunc.apply;
DWT_invert = DWTfunc.invert;

lambda_val = param_vals(1);
rwl1_reg = param_vals(2);
rwl1_mult = param_vals(3);

meas_func = MEAS_FUN{1};
Phit = meas_func.Phit; 

M = numel(MEAS_SIG(:, :, 1));
N2 = numel(DWT_apply(Phit(MEAS_SIG(:, :, 1))));

num_frames = size(MEAS_SIG, 3);
opts.tol = TOL;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run RWL1 on each frame

for kk = 1:num_frames
    tic
    if numel(MEAS_FUN) == 1
        meas_func = MEAS_FUN{1};
    elseif numel(MEAS_FUN) == num_frames
        meas_func = MEAS_FUN{kk};
    else
        error('You need either the same dynamics function for all time or one dynamics function per time-step!')
    end
    Phi  = meas_func.Phi;
    Phit = meas_func.Phit;
    
    weights = 1;
    for nn = 1:10
        % M-Step: Solve weighted BPDN
        Af = @(x) Phi(DWT_invert(x./weights));
        Ab = @(x) DWT_apply(Phit(x))./weights;
        A = linop_handles([M, N2], Af, Ab, 'R2R');
        res = solver_L1RLS(A, MEAS_SIG(:, :, kk), lambda_val, zeros(N2, 1), opts);
        res = res./weights;
        % E-stepL Reset the weights
        weights = rwl1_mult./(abs(res) + rwl1_reg);
        
        if rMSE_calc_opt == 1
            im_res = DWT_invert(res);
            temp_rMSE = sum(sum((im_res - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
            fprintf('Finished RW iteration %d. rMSE is %f.\n', nn, temp_rMSE)
        else
            fprintf('Finished RW iteration %d.\n', nn)
        end
    end

    if rMSE_calc_opt == 1
        im_res = DWT_invert(res);
        % Save reconstruction results
        vid_coef_rwcs(:, :, kk) = res;
        vid_recon_rwcs(:, :, kk) = im_res;
        vid_rMSE_rwcs(kk) = sum(sum((vid_recon_rwcs(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
        vid_PSNR_rwcs(kk) = psnr(real(vid_recon_rwcs(:, :, kk)), TRUE_VID(:, :, kk), 1);
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, TIME_ITER, vid_PSNR_rwcs(kk), vid_rMSE_rwcs(kk))
    else
        im_res = DWT_invert(res);
        % Save reconstruction results
        vid_coef_rwcs(:, :, kk) = res;
        vid_recon_rwcs(:, :, kk) = im_res;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

if (rMSE_calc_opt == 1)
    if nargout > 0
        varargout{1} = vid_coef_rwcs;
    end
    if nargout > 1
        varargout{2} = vid_recon_rwcs;
    end
    if nargout > 2
        varargout{3} = vid_rMSE_rwcs;
    end
    if nargout > 3
        varargout{4} = vid_PSNR_rwcs;
    end
    if nargout > 4
        for kk = 5:nargout
            varargout{kk} = [];
        end
    end
elseif (rMSE_calc_opt ~= 1)
    if nargout > 0
        varargout{1} = vid_coef_rwcs;
    end
    if nargout > 1
        varargout{2} = vid_recon_rwcs;
    end
    if nargout > 2
        for kk = 3:nargout
            varargout{kk} = [];
        end
    end
else
    error('How did you get here?')
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
