clc; clear; close all;
% Set the video information
videoSequence = 'news_qcif.yuv';

width  = 176;
height = 144;
nFrame = 300;
% nFrame = 90;

% Read the video sequence
[Y,U,V] = yuvRead(videoSequence, width, height ,nFrame); 
Y=double(Y);

Z=Y(:,:,1:300);

mr=0.70;
S = size(Z); 
Q = gen_W(S,mr);
B_Miss =  Z.*Q;

R1=120;
R2=120;
R3=60;
X=randn(144,176,300);
ps=[];
tic
for i=1:100
 Y=TuckerSample(X,R1,R2,R3);
X=Q.*Z+(~Q).*Y;
end
toc
