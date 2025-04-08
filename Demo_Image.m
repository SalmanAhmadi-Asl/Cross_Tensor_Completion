clc;clear all
% BB = imread('TextMask256.png');
% QS = double(BB>128);
%B=double(imread('house.bmp')); 
 B=double(imread('baboon.bmp'));
mr=0.90;
S = size(B); 
Q = gen_W(S,mr);

% Q(:,:,1)=QS;
% Q(:,:,2)=QS;
% Q(:,:,3)=QS;
% Q=double(B_Miss~=0);
 B_Miss =  B.*Q;
 

PSNR(B_Miss,B)
% SSIM(B_Miss,B)
% ssim(B_Miss,B)


X=randn(256,256,3);
% X=B_Miss;
ps=[];
%ms=[];

%%
R1=35;
R2=35;
R3=3;
tic

%%
for i=1:500
      Y=TuckerSamplesmooth(X,R1,R2,R3);
    X=B_Miss+(~Q).*Y;
    PSNR(X,B)
    %ps=[ps,PSNR(X,B)];
%     ms=[ms,ssim(X,B)];
    imshow(uint8(X))
    drawnow
end
%%
toc
%% %%%%%%%%%%% tubal
R1=35;
R2=35;
X=randn(256,256,3);
X=B_Miss;
ps=[];
sm=[];
tic
for i=1:150
Y=TubSamplsmooth(X,R1,R2);
X=B_Miss+(~Q).*Y;
imshow(uint8(X))
drawnow
ps=[ps,PSNR(X,B)];
sm=[sm,ssim(X,B)];
PSNR(X,B)
end
toc
