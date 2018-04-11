% if you use windows, use getenv('username')
if strcmp(getenv('USER'),'matthewoshaughnessy')
  mcodepath = '~/Documents/Classes/ECE 8803/project/PGM8803/code/foti2016sparse_matlab';
  pycodepath = '~/Documents/Classes/ECE 8803/project/fromauthors/proj/lvsglasso-admm/';
end

cd(mcodepath);
load('lvsglasso_inputs.mat');
lamS = 5e-4;
lamL = 1e-3;
save('lvsglasso_inputs.mat','fk','lamS','lamL');
copyfile('lvsglasso_inputs.mat',fullfile(pycodepath,'lvsglasso_inputs.mat'));

cd(pycodepath);
!python run_python_lvsglasso.py
copyfile('lvsglasso_outputs.mat',fullfile(mcodepath,'lvsglasso_outputs.mat'));
load('lvsglasso_outputs.mat');

cd(mcodepath);
%%
makevideo = true;
if makevideo
  v = VideoWriter('lvsglasso_out','MPEG-4');
  open(v);
end
for i = 1:size(S,1)
  Si = squeeze(S(i,:,:));
  Li = squeeze(L(i,:,:));
  subplot(211);
  imagesc(abs(Si));
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
sumS = squeeze(sum(S,1));
imagesc(abs(sumS));
title({'S (sum down freq dimension)', ...
  sprintf('||sumS||_0 = %d',sum(abs(Si(:))>1e-3))});
set(gca,'fontsize',16);
axis image; colorbar;