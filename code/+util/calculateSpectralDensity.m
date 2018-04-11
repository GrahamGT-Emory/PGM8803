% calculate estimated spectral density
% INPUTS
%   X     : T x p matrix of time series data
%   print : turn on debug print statements (default: true)
% OUTPUTS
%   fk : p x p x F matrix containing estimate of spectral density
%        (see Foti et al. 2016, Bach & Jordan 2004)
function [fk,df] = calculateSpectralDensity(X,print)


  % --- parse inputs ---
  [T,m] = size(X);
  nf = T; % todo
  if nargin < 2
    print = true;
  end
    

  % --- calculate DFTs ---
  % (manual method -- this is equivalent to the normalized fft below)
  %D = zeros(nf,m);
  %for i = 1:m
  %  for k = 1:T
  %    wk = 2*pi*(k-1)/T;
  %    D(k,i) = 1/sqrt(T)*sum(X(:,i).*exp(-1j*wk*(0:T-1).'));
  %  end
  %end
  D = (1/sqrt(T))*fft(X,nf);
  %assert(norm(D - D2, 'fro') < 1e-6);


  % --- calculate periodogram ---
  if print
    fprintf('Calculating periodogram...\n');
  end
  I = zeros(m,m,nf);
  for i = 1:nf
    I(:,:,i) = 1/sqrt(2*pi) * D(i,:).' * conj(D(i,:));
  end

  % --- smooth periodogram ---
  if print
    fprintf('Calculating optimal smoothing parameter...\n');
  end
  Wr = @(j,r) 1/(r*sqrt(2*pi)).*exp(-j.^2./(2*r^2));
  r_search = round(T^(1/5) : T^(3/10) : T^(4/5));
  aic_search = zeros(size(r_search));
  for ir = 1:length(r_search)
    r = r_search(ir);
    if print
      fprintf(' -> (value %d of %d) r = %d...', ir, length(r_search), r);
    end
    % calculate smoothed periodogram
    fhat = zeros(m,m,nf);
    for k = 1:T
      jj = (-T:T).';
      jj(k+jj < 1 | k+jj > T) = [];
      ww = shiftdim(Wr(jj,r),-2);
      fhat(:,:,k) = sum(ww.*I,3);
    end
    lw = 0;
    for ik = 1:T
      lw = lw - 1/2*( log(det(fhat(:,:,ik))) + ...
        trace(fhat(:,:,ik)\I(:,:,ik)) ) + ...
        -T*m/2*log(2*pi);
    end
    dfr = T/(r*sqrt(2*pi));
    aic_search(ir) = -lw + dfr/2*m^2;
    if print
      fprintf('AIC = %g.\n', aic_search(ir));
    end
  end


  % --- select best r using AIC criterion ---
  r_possible = 1:length(r_search);
  r_possible(abs(imag(aic_search)) > 1e-6) = [];
  [~,ir] = min(real(aic_search(r_possible))); % todo
  r_best = r_search(r_possible(ir));
  if print
    fprintf('Selected r = %d as parameter that minimizes AIC.\n', r_best);
  end
  r = r_best;


  % --- calculate smoothed periodogram with optimal value of r ---
  if print
    fprintf('Calculating smoothed periodogram with optimal value of r...\n');
  end
  fhat = zeros(m,m,nf);
  for k = 1:T
    jj = (-T:T).';
    jj(k+jj < 1 | k+jj > T) = [];
    ww = shiftdim(Wr(jj,r),-2);
    fhat(:,:,k) = sum(ww.*I,3);
  end


  % --- subsample the smoothed periodogram ---
  H = 4*T/(r*sqrt(2*pi));
  subsamp_factor = round(H); %round(T/H); -- matt fixed 04/11/18
  if print
    fprintf(' -> Optimal subsampling factor: %d.\n', subsamp_factor);
  end
  df = T/(r*sqrt(2*pi));
  fk = fhat(:,:,1:subsamp_factor:end);
  if print
    fprintf(' -> Output spectral density matrix size: %dx%dx%d.\n', ...
      m, m, size(fk,3));
  end
  if print
    fprintf('Done!\n');
  end


end
