% construct Astar
%% My attempt
% Astar = [A B; C D] 
p = 50;
r = 5;
T = 1000;
snr = 2;
A = zeros(p,p);
for i = 1:p
    for j = 1:p
        if i==j     
            A(i,j) = .2;
        end
    end
end   
for i = 1:p
    c = round(p*rand)+1;
    if round(rand)==1
        A(i, c) = round(rand)-.5;
    end
    c = round(p*rand)+1;
    if round(rand)==1
        A(i, c) = round(rand)-.5;
    end
end

B = zeros(p,r);
for i = 1:p
    for j = 1:r
        B(i,j) = normrnd(0,2);
    end
end
for i = 1:r
    count = 0;
    while count<p/5
        B(round(p*rand)+1,i) =0;
        count = count+1;
    end
end
if size(B,1)>50, B(size(B,1),:)=[]; end

C = zeros(r,p); % done
D = zeros(r,r); 
for i = 1:r
    for j = 1:r
        if i==j     
            D(i,j) = randn;
        end
    end
end            % done

Astar = [A B; C D];
Astar = Astar / max(eig(Astar));



%% Theirs
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