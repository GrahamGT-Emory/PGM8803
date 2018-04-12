%%
% if you use windows, use getenv('username')
if strcmp(getenv('USER'),'matthewoshaughnessy')
  mcodepath = '~/Documents/Classes/ECE 8803/project/PGM8803/code/rwlvsglasso';
  pycodepath = '~/Documents/Classes/ECE 8803/project/fromauthors/proj/lvsglasso-admm/';
end


%% run without dynamics

cd(mcodepath);
load('rwlvsglasso_inputs.mat');
symA = abs(A)+abs(A)' > 0;
LamS = 6e-3*ones(size(fk));
lamL = 3e-2;
save('rwlvsglasso_inputs.mat','fk','LamS','lamL','A','symA');
copyfile('rwlvsglasso_inputs.mat',fullfile(pycodepath,'rwlvsglasso_inputs.mat'));

cd(pycodepath);
!python run_python_rwlvsglasso.py
copyfile('rwlvsglasso_outputs.mat',fullfile(mcodepath,'rwlvsglasso_outputs.mat'));
cd(mcodepath);
load('rwlvsglasso_outputs.mat');

allS = squeeze(all(abs(S) > 1e-3,1));
f1_score = util.evaluateGraph(allS,symA);


%% run with dynamic estimate

cd(mcodepath);
load('rwlvsglasso_inputs.mat');
symA = abs(A)+abs(A)' > 0;
prior = (1-symA);
LamS = 8e-3*(prior + 1e-4);
lamL = 3e-2;
save('rwlvsglasso_inputs.mat','fk','LamS','lamL','A','symA');
copyfile('rwlvsglasso_inputs.mat',fullfile(pycodepath,'rwlvsglasso_inputs.mat'));

cd(pycodepath);
!python run_python_rwlvsglasso.py
copyfile('rwlvsglasso_outputs.mat',fullfile(mcodepath,'rwlvsglasso_outputs.mat'));
cd(mcodepath);
load('rwlvsglasso_outputs.mat');

allS = squeeze(all(abs(S) > 1e-3,1));
f1_score = util.evaluateGraph(allS,symA);


%%
makevideo = false;
if makevideo
  v = VideoWriter('lvsglasso_out','MPEG-4');
  open(v);
end
for i = 1:size(S,1)
  Si = squeeze(S(i,:,:));
  Li = squeeze(L(i,:,:));
  subplot(211);
  imagesc(abs(Si)); colorbar;
  title({'Recovered sparse component', ...
    sprintf('(frame %d -- ||S_i||_0 = %d)',i,sum(abs(Si(:))>1e-3))});
  set(gca,'fontsize',16); axis image;
  subplot(212);
  imagesc(abs(Li));
  title({'Recovered low-rank component', sprintf('(frame %d -- rank %d)',i,rank(squeeze(L(i,:,:))))});
  set(gca,'fontsize',16); axis image;
  drawnow;
  if makevideo
    writeVideo(v,getframe(gcf));
  end
end
if makevideo
  close(v);
end


%%
clf;
allS = squeeze(all(abs(S) > 1e-3,1));
sumS = squeeze(sum(S(13,:,:),1));
sumSnodiag = sumS - diag(diag(sumS));
subplot(211);
imagesc(allS);
title(sprintf('K_Y (||K_Y||_0 = %d)',sum(allS(:))));
set(gca,'fontsize',16);
axis image;
subplot(212);
symA = abs(A)+abs(A)' > 0;
imagesc(symA);
title(sprintf('A + A^T (||A+A^T||_0 = %d)',sum(symA(:))));
axis image;
set(gca,'fontsize',16);


%%
clf;
allS = squeeze(all(abs(S) > 1e-3,1));
sumS = squeeze(sum(S(13,:,:),1));
imshowpair(symA,allS);
axis image;
set(gca,'fontsize',16);

