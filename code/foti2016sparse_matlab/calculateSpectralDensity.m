% calculate estimated spectral density

% --- parameters ---
F = T;
useFFT = true;
ww = hann(F); % TODO - in freq domain?, also, tune length


% --- calculate periodogram I(wk) ---
tic;
kk = 0:T-1;
tt = 0:T-1;
I = zeros(p,p,F);
S = zeros(p,p,F);
for i = 1:p
  % calculate DFT dd (T x 1)
  if useFFT
    dd = 1/sqrt(T)*shiftdim(fft(X(i,:)),1);
  else
    dd = zeros(T,1);
    for ik = 1:T
      k = kk(ik);
      for it = 1:T
        t = tt(it);
        wk = 2*pi*k/T;
        dd(ik) = dd(ik) + 1/sqrt(T)*X(i,it)*exp(-1j*wk*t);
      end
    end
  end
  % calculate periodogram
  I(:,:,i) = 1/(2*pi)*dd.*conj(dd);
end


% --- calculate smoothed periodogram
for i = 1:p
  S(:,:,i) = ww.*I(:,:,i);
end




