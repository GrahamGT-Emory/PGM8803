function Psi = calculateSpectralDensity(X,nf,ww)
% calculate estimated spectral density
% INPUTS
%   X   : T x p matrix of time series data
%   nf  : (optional) # frequency bins to use (default: T)
%   ww  : (optional) nf x 1 window (default: hanning)
% OUTPUTS
%   Psi : p x p x F matrix of est. cross-spectral density
%         notation from Foti et al. 2016:
%          - Psi[f]      is Psi(:,:,f)
%          - Psi[.]_{ij} is Psi(i,j,:)

  % --- parse inputs ---
  [T,p] = size(X);
  if nargin < 2
    nf = T;
  end
  if nargin < 3
    ww = hann(round(nf));
  end

  % --- calculate DFTs ---
  D = zeros(nf,p);
  for i = 1:p
    for k = 1:T
      D(k,i) = 1/T*sum(X(:,1)*exp(-1j*(k-1)*(0:T-1)));
    end
  end
  D2 = (1/sqrt(T))*fft(X,nf);

  % --- calculate smoothed periodogram ---
  Psi = zeros(p,p,nf);
  for i = 1:p
    for j = 1:p
       st = squeeze(D(i,:) .* conj(D(j,:)))./(2*pi);
       st = conv(st, ww/sum(ww),'same');
       Psi(i,j,:) = reshape(st, [1 1 nf]);
    end
  end

end