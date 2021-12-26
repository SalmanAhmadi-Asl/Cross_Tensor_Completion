clc;clear
load('Yale_64x64.mat')
B=reshape(fea,[11,15,64,64]);
imshow(uint8(squeeze(B(1,2,:,:))));
X_miss=B;
mr=0.70;
S = size(B); 
Q = gen_W(S,mr);
 X_miss=B.*Q;
X_miss=B.*Q;
% subplot(1,7,1)
% imshow(uint8(squeeze(X_miss(1,1,:,:))))

R1=11;
R2=15;
R3=50;
R4=50;
X=randn(11,15,64,64);
ps=[];
tic
for i=1:200

%  Y=double(FSTD(X,R1,R2,R3,R4));
Y=TuckerSample4(X,R1,R2,R3,R4);
X=X_miss+(~Q).*Y;
ER=X-B;
norm(ER(:))/norm(B(:))
end
toc
subplot(2,6,1)
imshow(uint8(squeeze(X(1,1,:,:))))
subplot(2,6,2)
imshow(uint8(squeeze(X(1,14,:,:))))
subplot(2,6,3)
imshow(uint8(squeeze(X(5,1,:,:))))
subplot(2,6,4)
imshow(uint8(squeeze(X(10,7,:,:))))
subplot(2,6,5)
imshow(uint8(squeeze(X(8,11,:,:))))
subplot(2,6,6)
imshow(uint8(squeeze(X(11,11,:,:))))

subplot(2,6,7)
imshow(uint8(squeeze(B(1,1,:,:))))
subplot(2,6,8)
imshow(uint8(squeeze(B(1,14,:,:))))
subplot(2,6,9)
imshow(uint8(squeeze(B(5,1,:,:))))
subplot(2,6,10)
imshow(uint8(squeeze(B(10,7,:,:))))
subplot(2,6,11)
imshow(uint8(squeeze(B(8,11,:,:))))
subplot(2,6,12)
imshow(uint8(squeeze(B(11,11,:,:))))