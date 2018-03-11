p = 8;
N = 256;

S_true = zeros(p);
S_true(3,1) = 1;
S_true(5,2) = 1;
S_true = 1/2*(S_true + S_true');

X = S_true*randn(p,N);

S_est1 = 1/(N-1)*(X*X');
S_est2 = cov(X.');