% calculate estimated spectral density

% --- parameters ---
F = T;
useFFT = true;
ww = hann(F); % TODO - tune length

% --- calculate DFTs ---
D = 1/sqrt(T)*fft(X,[],2);

% --- calculate periodogram ---
I = zeros(p,p,F);
for i = 1:p
  for j = 1:p
    I(i,j,:) = D(i,:) .* conj(D(j,:));
  end
end

% --- calculate smoothed periodogram ---
S = I .* reshape(ww/sum(ww),[1 1 F]);
Psi = S;

% notation from Foti et al. 2016:
%  - Psi[f]      is Psi(:,:,f)
%  - Psi[.]_{ij} is Psi(i,j,:)