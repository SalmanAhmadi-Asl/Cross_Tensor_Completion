function [Y]=MahonSamplsmooth(X,R1,R2)
X=double(X);
N_m=size(X);

r=randsample(N_m(3),R1);

C=X(:,:,r);

for i=1:size(r)
for j=1:N_m(2)
C(:,j,i)=smooth(squeeze(C(:,j,i)));
end
end

for i=1:R2
r1 = randi([1 N_m(1)],1,1);
r2 = randi([1 N_m(2)],1,1);
rr1(i)=r1;
rr2(i)=r2;
R(i,:)=smooth(X(r1,r2,:));
end


for i=1:R2
for j=1:R1
W(i,j)=X(rr1(i),rr2(i),r(j));
end
end

D1=zeros(size(W,2),size(W,2));
D2=zeros(size(W,1),size(W,1));
for i=1:size(W,2)
D1(i,i)=1/sqrt(size(W,1));
end

for i=1:size(W,1)
D2(i,i)=1/sqrt(size(W,2));
end

RR=D1*pinv(D2*W*D1)*D2;
Y=tmprod(C,(RR*R)',3);

end