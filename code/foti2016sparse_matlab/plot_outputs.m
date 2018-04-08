% add path to authors' code -- use getenv('username') on windows
if strcmpi(getenv('USER'),'matthewoshaughnessy')
  addpath('~/Documents/Classes/ECE 8803/project/fromauthors/proj/lvsglasso-admm/');
end

load lvsglasso_inputs
load lvsglasso_outputs

S_act = info.S_Input > 0.01;
Sout_thresh = squeeze(~all(abs(S) < 1,1));

subplot(211); imagesc(S_act); title('actual');
subplot(212); imagesc(Sout_thresh); title('output');

f1 = evaluateGraph(Sout_thresh, S_act);