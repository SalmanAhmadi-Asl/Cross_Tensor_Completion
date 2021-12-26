clc;clear
% load mri;
% montage(D,map)
% B=double(squeeze(D));
% imshow(uint8(squeeze(B(:,:,1))));
V = niftiread('brain.nii');
B=double(V);
mr=0.40;
S = size(B); 
Q = gen_W(S,mr);
% imshow(uint8(squeeze(X_miss(:,:,1))))

X_miss=B.*Q;

% span=5;
% smoothmethod='moving';

R1=160;
R2=160;
R3=21;
X=randn(256,256,21);
ps=[];
tic
for i=1:250

% Y=double(FSTD(X,R1,R2,R3));
Y=TuckerSample(X,R1,R2,R3);
X=X_miss+(~Q).*Y;

end
toc
subplot(1,3,1)
imshow(uint8(squeeze(X(:,:,6))))
subplot(1,3,2)
imshow(uint8(squeeze(B(:,:,6))))
subplot(1,3,3)
imshow(uint8(squeeze(X_miss(:,:,6))))
hold on
ps=[];
sm=[];
for i=1:21
ps=[ps,PSNR(X(:,:,i),B(:,:,i))];
sm=[sm,ssim(X(:,:,i),B(:,:,i))];
end
x=1:21;
plot(x,ps)
xlim([1 21])


