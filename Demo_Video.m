clc; clear; close all;
% Set the video information
% videoSequence = 'sampleQCIF.yuv';
videoSequence = 'news_qcif.yuv';
%  videoSequence = 'foreman_qcif.yuv'; 
% videoSequence = 'carphone_qcif.yuv'; 
%videoSequence='bridge-close_qcif.yuv';
%videoSequence = 'stefan_cif.yuv'; 
%videoSequence = 'foreman_qcif.yuv'; 
%videoSequence ='akiyo_qcif.yuv';
% grandma_qcif
% bridge-far_cif
% grandma_qcif
%bridge-close_cif
%akiyo_qcif

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

% B_Miss(:,:,)
% for i=1:30
% imshow(uint8(B_Miss(:,:,i)))
% end

% PSNR(B_Miss,B)
% SSIM(B_Miss,B)

R1=120;
R2=120;
R3=60;
X=randn(144,176,300);
ps=[];
tic
for i=1:100
% r = randi([1 S12],1,R1);
% %r=randsample(S12,R1);
% D{1}=C{1}(:,r);
% QQ{1}=pinv(D{1});
% 
% r = randi([1 S22],1,R2);
% %r=randsample(S22,R2);
% D{2}=C{2}(:,r);
% QQ{2}=pinv(D{2});
% 
% r = randi([1 S32],1,R3);
% %r=randsample(S32,R3);
% D{3}=C{3}(:,r);
% QQ{3}=pinv(D{3});
% 
% %S2=((ttensor(tensor(X),pinv(D{1}),pinv(D{2}),pinv(D{3}))));
% S2=tmprod(B_Miss,QQ,[1,2,3]);
% %S2=lmlragen(Q,double(X_Noisy));
% Y=tmprod(S2,D,[1,2,3]);
%Y=double(FSTD(X,R1,R2,R3));
 Y=TuckerSample(X,R1,R2,R3);
X=Q.*Z+(~Q).*Y;
% ps=[ps,PSNR(X,B)];

% C{1}=double(tenmat((X),[1]));
% C{2}=double(tenmat((X),[2]));
% C{3}=double(tenmat((X),[3]));

end
toc

ps=[];
for i=1:300
%imshow(uint8(B_Miss(:,:,i)))
ps=[ps,PSNR(B_Miss(:,:,i),Z(:,:,i))];
end
% figure(2)
plot(ps)
ylim([6,15])

 hold on
% figure(3)
ps=[];
for i=1:300
%imshow(uint8(X(:,:,i)))
ps=[ps,PSNR(X(:,:,i),Z(:,:,i))];
end
% figure(4)
plot(ps)
ylim([20,30])

TT=[];
TT2=[];
figure(5)
for i=1:5
subplot(3,5,i)
imshow(uint8(B_Miss(:,:,i*5)))
TT=[TT,PSNR(B_Miss(:,:,i),Z(:,:,i))];
TT2=[TT2,ssim(B_Miss(:,:,i),Z(:,:,i))];
end
ZZ=[];
ZZ2=[];
    j=1;
for i=6:10
subplot(3,5,i)
imshow(uint8(X(:,:,j*5)))
ZZ=[ZZ,PSNR(B_Miss(:,:,j*5),Z(:,:,j*5))];
ZZ2=[ZZ2,ssim(B_Miss(:,:,j*5),Z(:,:,j*5))];
j=j+1;
end

Sn=[10,40,90,150,260];
for i=1:5
subplot(1,5,i)
imshow(uint8(Z(:,:,Sn(i))))
end
%%%%%%%%%%%%%tubal%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R1=35;
R2=45;
X=randn(144,176,300);
ps=[];
sm=[];
tic
for i=1:100
Y=TubSampl2(X,R1,R2);
X=Q.*Z+(~Q).*Y;
PSNR(B_Miss(:,:,i),Z(:,:,i))
i
end
toc
ps=[];
for i=1:30
imshow(uint8(B_Miss(:,:,i)))
ps=[ps,PSNR(B_Miss(:,:,i),Z(:,:,i))];
end
figure(2)
plot(ps)

hold on
%figure(3)
ps=[];
for i=1:300
imshow(uint8(X(:,:,i)))
%ps=[ps,PSNR(X(:,:,i),Z(:,:,i))];
end
%figure(4)
plot(ps)
ylim([25,50])


figure(5)
for i=1:5
subplot(2,5,i)
imshow(uint8(B_Miss(:,:,i*5)))
end
j=1;
for i=6:10
    
subplot(2,5,i)
imshow(uint8(X(:,:,j*5)))
j=j+1;
end
%%%%%%%%%%%%%%%%%Mahoney
%%For all images we used 23 slices and 1200 tubes except he 90% missing case for
%%which we have used 13 sllices and 1500 tubes in each iteration

R1=90;
R2=1500;
X=randn(144,176,300);
ps=[];
sm=[];
tic
for i=1:100
Y=MahonSampl(X,R1,R2);
X=Q.*Z+(~Q).*Y;
% ps=[ps,PSNR(Y,B)];
% sm=[sm,SSIM(Y,B)];
end
toc


ps=[];
for i=1:300
imshow(uint8(X(:,:,i)))
ps=[ps,PSNR(B_Miss(:,:,i),Z(:,:,i))];
end
figure(2)
plot(ps)

hold on
%  figure(2)
ps=[];
for i=1:300
%imshow(uint8(X(:,:,i)))
ps=[ps,PSNR(X(:,:,i),Z(:,:,i))];
end
% figure(4)
plot(ps)
ylim([8,45])

figure(5)
for i=1:5
subplot(2,5,i)
imshow(uint8(B_Miss(:,:,i*5)))
end

   j=1;
for i=6:10
subplot(2,5,i)
imshow(uint8(X(:,:,j*5)))
j=j+1;
end
%%%%%%%%%%%%%%TRLF
r=5*ones(1,3); % TR-rank 
maxiter=300; % maxiter 300~500
tol=1e-6; % 1e-6~1e-8
Lambda=5; % usually 1~10
ro=1; % 1~1.5
K=1e0; % 1e-1~1e0 
tic
[X,~,~]=TRLRF(Z,Q,r,maxiter,K,ro,Lambda,tol);
toc

ps=[];
for i=1:30
imshow(uint8(B_Miss(:,:,i)))
ps=[ps,PSNR(B_Miss(:,:,i),Z(:,:,i))];
end
figure(2)
plot(ps)

% figure(3)
hold on
ps=[];
for i=1:300
%imshow(uint8(X(:,:,i)))
ps=[ps,PSNR(X(:,:,i),Z(:,:,i))];
end
% figure(4)
plot(ps)
ylim([8,45])

figure(5)
for i=1:5
subplot(2,5,i)
imshow(uint8(B_Miss(:,:,i*5)))
end

   j=1;
for i=6:10
subplot(2,5,i)
imshow(uint8(X(:,:,j*5)))
j=j+1;
end

%%%%%%%%%%%%%%%WOPT
N=3;
r_select=6;
 maxiter_trwopt=100;
    tic
    X=WTR(Z,Q,r_select*ones(1,N),maxiter_trwopt);
    toc
%     imshow(uint8(X))  
hold on
ps=[];
for i=1:300
%imshow(uint8(X(:,:,i)))
ps=[ps,PSNR(X(:,:,i),Z(:,:,i))];
end
% figure(4)
plot(ps)
ylim([20,45])

figure(5)
for i=1:5
subplot(2,5,i)
imshow(uint8(B_Miss(:,:,i*5)))
end

   j=1;
for i=6:10
subplot(2,5,i)
imshow(uint8(X(:,:,j*5)))
j=j+1;
end
%%%%%%%%%%%%%%TRALS
% 
% figure()
% imshow(uint8(Data_Missing));
% saveas(gcf, strcat(filename,['/Einstein_ObservingRate' num2str(ObserveRatio) '.pdf']));
%% Complete Missing Data By TR
% TR Parameter
Omega=Q;
Data=Z;
Data_Size = size(Data);
Data_Missing = reshape(T2V(Data).* T2V(Omega), Data_Size);
% figure()
% imshow(uint8(Data_Missing));
Data_Size = size(Z);
%Reshape_Dim   = [4,4,16,4,4,16,3];
Reshape_Dim = [16,8,20,6,33,15];
r = 10;
para_TR.max_tot = 10^-4;   
para_TR.max_iter= 10;     
para_TR.disp = 1;
para_TR.r = ones(length(Reshape_Dim), 1)*r;   % Tensor Ring Rank
%para_TR.r(end)=1; % if TT

tic
% TR Completion
Utr = Completion_TR(reshape(Data_Missing, Reshape_Dim), reshape(Omega, Reshape_Dim), para_TR);

%Utr = TR_comp_Incr(reshape(Data_Missing, Reshape_Dim), reshape(Omega, Reshape_Dim), para_TR);
Data_Recover_TR = reshape(Ui2U(Utr), Data_Size);

%add the constraint here
Data_Recover_TR(Omega(:)==1) = Data(Omega(:)==1);
toc

% 
%    mu = 1e-4;
%    for k=1:100
%  [X,tnnX,trN] = prox_tnn(Utr{1},1/mu); 
%    end

% Plot recovered images
% figure()
% imshow(uint8(Data_Recover_TR));
% saveas(gcf, strcat(filename,['/Einstein_Completion(TR)Rank' num2str(r) 'Err' num2str(norm(T2V(Data_Recover_TR-Data))/norm(T2V(Data))) '.pdf']));
% imwrite(uint8(Data_Recover_TR),'barbara_TR.png');
% imwrite(uint8(Data_Missing),'lena_missing1180.png')

% RSE = norm(Data(:) - Data_Recover_TR(:))/norm(Data(:)); 
% PSNR= PSNR(Data,Data_Recover_TR)
% SSIM= SSIM(Data,Data_Recover_TR)
% img_set= Data_Recover_TR;

% subplot(1,3,1)
% imshow(uint8(Data))
% subplot(1,3,2)
% imshow(uint8(Data_Missing))
% subplot(1,3,3)
% imshow(uint8(Data_Recover_TR))


% figure(3)
hold on
ps=[];
for i=1:300
% imshow(uint8(Data_Recover_TR(:,:,i)))
ps=[ps,PSNR(Data_Recover_TR(:,:,i),Data(:,:,i))];
end
% figure(4)
plot(ps)
ylim([20,48])

figure(5)
for i=1:5
subplot(2,5,i)
imshow(uint8(Data_Missing(:,:,i*5)))
end

   j=1;
for i=6:10
subplot(2,5,i)
imshow(uint8(Data_Recover_TR(:,:,j*5)))
j=j+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%SPC
clc;clear all
%X =double(imread('house.bmp')); 
X=Z;
% mr=0.70;
% S = size(X); 
% Q = gen_W(S,mr);
B_Miss=Q.*X;
% for k=1:3
% for i=1:256
%    for j=1:256
%    if Q(i,j,k)==1
%        Q2(i,j,k)=1;
%    end
%    end
% end
% end
% 
% Q=Q2;
% B=X;

X0 =  double(X).*Q;
II = size(X0);

%% set missing indexes
Q = (X0 ~= 0);
% Q=(X0~=0);
T = zeros(II);
T(Q) = double(X0(Q));

%% hyperparameters and run SPC

%TVQV    = 'qv';        % 'tv' or 'qv' ;
%rho     = [0.01 0.01 0]; % smoothness (0.1 - 1.0) for 'qv' and (0.01 - 0.5) for 'tv' is recommended.

TVQV    = 'qv';        % 'tv' or 'qv' ;
rho     = [0.5 0.5 0]; % smoothness (0.1 - 1.0) for 'qv' and (0.01 - 0.5) for 'tv' is recommended.
K       = 10;          % Number of components which are updated in one iteration. (typically 10)
SNR     = 25;          % error bound
nu      = 0.01;        % threshold for R <-- R + 1.
maxiter = 200;        % maximum number of iteration
tol     = 1e-5;        % tolerance
out_im  = 1;           % you can monitor the process of 'image' completion if out == 1. 'saved' directory is necessary to save the individual rank images.

tic
[X W G U histo histo_R] = SPCF(T,Q,TVQV,rho,K,SNR,nu,maxiter,tol,out_im);
toc

% figure(3)
hold on
ps=[];
for i=1:300
% imshow(uint8(X(:,:,i)))
ps=[ps,PSNR(X(:,:,i),Z(:,:,i))];
end
% figure(4)
plot(ps)
ylim([20,48])

figure(5)
for i=1:5
subplot(2,5,i)
imshow(uint8(B_Miss(:,:,i*5)))
end

   j=1;
for i=6:10
subplot(2,5,i)
imshow(uint8(X(:,:,j*5)))
j=j+1;
end