% constructs data for VAR(1) synthetic data experiment in Foti et al.
% 2016, section 6.1
%
% Matt O'Shaughnessy, Georgia Tech, 9 March 2018

% params
p = 69;
r = 5;
T = 512;
snr = 2;


% construct Astar
A = diag(0.2*ones(p,1));
for i = 1:p
  ind = randperm(p,2);
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

