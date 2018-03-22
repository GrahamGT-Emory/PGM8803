function [varargout] = BPDN_video(varargin)

% [vid_coef_cs, vid_recon_cs, vid_rMSE_cs, vid_PSNR_cs] = ...
%           BPDN_video(MEAS_SIG, MEAS_SEL, lambda_val, TOL, TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% lambda_val: Scalar value for the BPDN sparsity tradeoff
% TOL:        Scalar value for the tolerance in the TFOCS solver
% DWTfunc:    Wavelet transform (sparsity basis)
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate errors)
% 
%    The outputs are:
% 
% vid_coef_cs:  Nx1xT array of inferred sparse coefficients
% vid_recon_cs: Sqrt(N)xSqrt(N)xT array of the recovered video sequence
% vid_rMSE_cs:  Tx1 array of rMSE values for the recovered video
% vid_PSNR_cs:  Tx1 array of PSNR values for the recovered video
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
lambda_val = varargin{3};
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

meas_func = MEAS_FUN{1};
Phit = meas_func.Phit; 

M = numel(MEAS_SIG(:, :, 1));
N2 = numel(DWT_apply(Phit(MEAS_SIG(:, :, 1))));

num_frames = size(MEAS_SIG, 3);
opts.tol = TOL;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run BPDN on each frame

for kk = 1:num_frames
    tic
    % Set up the measurement function
    if numel(MEAS_FUN) == 1
        meas_func = MEAS_FUN{1};
    elseif numel(MEAS_FUN) == num_frames
        meas_func = MEAS_FUN{kk};
    else
        error('You need either the same dynamics function for all time or one dynamics function per time-step!')
    end
    Phi  = meas_func.Phi;
    Phit = meas_func.Phit;
    % Set up A and At for TFOCS
    Af = @(x) Phi(DWT_invert(x));
    Ab = @(x) DWT_apply(Phit(x));
    A = linop_handles([M, N2], Af, Ab, 'R2R');
    
    % Solve the BPDN objective
    res = solver_L1RLS( A, MEAS_SIG(:, :, kk), lambda_val, zeros(N2, 1), opts );
    im_res = DWT_invert(res);
    
    % Save reconstruction results
    vid_coef_cs(:, :, kk) = res;
    vid_recon_cs(:, :, kk) = im_res;
    if rMSE_calc_opt == 1
        vid_rMSE_cs(kk) = sum(sum((vid_recon_cs(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
        vid_PSNR_cs(kk) = psnr(real(vid_recon_cs(:, :, kk)), TRUE_VID(:, :, kk), 1);
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, TIME_ITER, vid_PSNR_cs(kk), vid_rMSE_cs(kk))
    else
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds.\n', kk, num_frames, TIME_ITER)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

if (rMSE_calc_opt == 1)
    if nargout > 0
        varargout{1} = vid_coef_cs;
    end
    if nargout > 1
        varargout{2} = vid_recon_cs;
    end
    if nargout > 2
        varargout{3} = vid_rMSE_cs;
    end
    if nargout > 3
        varargout{4} = vid_PSNR_cs;
    end
    if nargout > 4
        for kk = 5:nargout
            varargout{kk} = [];
        end
    end
elseif (rMSE_calc_opt ~= 1)
    if nargout > 0
        varargout{1} = vid_coef_cs;
    end
    if nargout > 1
        varargout{2} = vid_recon_cs;
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
