% --- get all S&P 500 data ---
if exist('snp500_data_all.mat','file')
  load('snp500_data_all.mat');
else
  [~,~,snp500_table] = xlsread('snp500_table.xlsx');
  tickers = snp500_table(:,1).';
  data = hist_stock_data('01012000','12312012',tickers,'d');
end

%%
% --- remove short-lived tickers ---
ndates = zeros(length(data),1);
for i = 1:length(data)
  ndates(i) = length(data(i).Date);
end
ndates_all = max(ndates);
igood = ndates == ndates_all;
data_good = data(igood);

% --- get category for each ---
for i = 1:length(data_good)
  ticker = data_good(i).Ticker;
  ind = strcmp(snp500_table(:,1).',ticker);
  data_good(i).Name = snp500_table{ind,2};
  data_good(i).Sector = snp500_table{ind,3};
  data_good(i).SubIndustry = snp500_table{ind,4};
end

% --- calculate ratios ---
for i = 1:length(data_good)
  data_good(i).Ratio = log( data_good(i).AdjClose(2:end) ./ data_good(i).AdjClose(1:end-1) );
end

%%
snp = data_good;
save('snp500_data','snp');

