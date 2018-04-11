% if you use windows, use getenv('username')
if strcmp(getenv('USER'),'matthewoshaughnessy')
  mcodepath = '~/Documents/Classes/ECE 8803/project/PGM8803/code/foti2016sparse_matlab';
  pycodepath = '~/Documents/Classes/ECE 8803/project/fromauthors/proj/lvsglasso-admm/';
end

cd(mcodepath);
load('lvsglasso_inputs.mat');
lamS = 1.22e-2;
lamL = 2e-2;
save('lvsglasso_inputs.mat','fk','invfk','lamS','lamL');
copyfile('lvsglasso_inputs.mat',fullfile(pycodepath,'lvsglasso_inputs.mat'));

cd(pycodepath);
!python run_python_lvsglasso.py
copyfile('lvsglasso_outputs.mat',fullfile(mcodepath,'lvsglasso_outputs.mat'));
load('lvsglasso_outputs.mat');

cd(mcodepath);
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

