p = 50;
r = 5;
T = 1000;
snr = 200;


% construct Astar
A = diag(0.2*ones(p,1));
for i = 1:p
  ind = randperm(p,2);
  while any(ind == i)
      ind = randperm(p,2);
  end
  A(i,ind) = round(rand(1,2))-1/2;
end
B = 2*randn(p,r);
for i = 1:r
  ind = randperm(p,round(0.2*p));
  B(ind,i) = 0;
end
C = zeros(r,p);
D = diag(randn(r,1));
Astar = [A B; C D];
Astar = Astar / max(eig(Astar));
% calculate x and u
sigma = norm(Astar,2) / snr; % todo - check this is spectral norm
XU = zeros(p+r,T);
XU(:,1) = randn(p+r,1);
for t = 2:T
  e = sigma*randn(p+r,1);
  XU(:,t) = Astar*XU(:,t-1) + e;
end
X = XU(1:p,:);
U = XU(p+1:end,:);


C_actual = (1/(T-1))*(XU*XU');
S = inv(C_actual);
% To avoid numerical errors, sets low values to 0 for plotting graphs
threshold = quantile(real(S(:)),[0.90]);
num_samples = 1000;
C_sample = (1/(num_samples-1))*(X(:,1:num_samples)*X(:,1:num_samples)');
targets = abs(S(1:p, 1:p)) > threshold;


specden = permute(calculateSpectralDensity(XU,32),[3 1 2]);


save('lvsglasso_inputs.mat','specden');


