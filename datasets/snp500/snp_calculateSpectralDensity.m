% calculate spectral density matrix for S&P 500 stock data

%%
load snp500_data

%% extract data matrix
nVariables = length(snp);
nTime = length(snp(1).Ratio);
X = zeros(nTime,nVariables);
for i = 1:nVariables
  X(:,i) = snp(i).Ratio;
end

%% create spectral density matrix
cpsd(