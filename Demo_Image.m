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
PSNR(X,B)

figure(2)
x=1:50;
plot(x,ps)
hold on
PSNR(X,B)
SSIM(X,B)
ssim(X,B)

plot(ps)
hold on
% figure(2)
% subplot(1,3,1)
figure(1)
imshow(uint8(B))
% subplot(1,3,2)
figure(2)
imshow(uint8(B_Miss))
% subplot(1,3,3)
figure(3)
imshow(uint8(X))
PSNR(X,B)
SSIM(X,B)

%%%%%%%%%%%%%tubal
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
% PSNR(X,B)
% SSIM(X,B)
% ssim(X,B)
x=1:150;
plot(x,ps)
hold on
subplot(1,4,1)
imshow(uint8(X))
plot(ps)
subplot(1,2,2)
plot(sm)
figure(2)
subplot(1,3,1)
imshow(uint8(B))
subplot(1,3,2)
imshow(uint8(B_Miss))
subplot(1,3,3)
imshow(uint8(X))
PSNR(X,B)
SSIM(X,B)
%%%%%%%%%%%%%%%%%Mahoney
%%For all images we used 23 slices and 1200 tubes except the 90% missing case for
%%which we have used 13 sllices and 1500 tubes in each iteration
B_Miss2=reshape(B_Miss,[32,64,96]);
B=reshape(B,[32,64,96]);
Q=reshape(Q,[32,64,96]);

R1=40;
R2=3000;
X=randn(32,64,96);
X=B_Miss2;
ps=[];
sm=[];
tic
for i=1:500
Y=MahonSamplsmooth(X,R1,R2);
X=B_Miss2+(~Q).*Y;
X_RE=reshape(X,[256,256,3]);
B_RE=reshape(B,[256,256,3]);
PSNR(X_RE,B_RE)
imshow(uint8(X_RE))
drawnow
% ps=[ps,PSNR(Y,B)];
% sm=[sm,SSIM(Y,B)];
end
toc
X=reshape(X,[256,256,3]);
B=reshape(B,[256,256,3]);
B_Miss2=reshape(B_Miss,[256,256,3]);
SSIM(X,B)
PSNR(X,B)
subplot(1,3,1)
imshow(uint8(B))
subplot(1,3,2)
imshow(uint8(B_Miss))
subplot(1,3,3)
imshow(uint8(X))
%%%%%%%%%%%%%%TRLF
r=5*ones(1,3); % TR-rank 
maxiter=300; % maxiter 300~500
tol=1e-6; % 1e-6~1e-8
Lambda=5; % usually 1~10
ro=1; % 1~1.5
K=1e0; % 1e-1~1e0 
tic
[X,~,~]=TRLRF(B,Q,r,maxiter,K,ro,Lambda,tol);
toc
    imshow(uint8(X))
PSNR(X,B)
SSIM(X,B)  
subplot(1,3,1)
imshow(uint8(B))
subplot(1,3,2)
imshow(uint8(B_Miss))
subplot(1,3,3)
imshow(uint8(X))
PSNR(X,B)
SSIM(X,B)
%%%%%%%%%%%%%%%WOPT
N=3;
r_select=10;
 maxiter_trwopt=100;
    tic
    X=WTR(B,Q,r_select*ones(1,N),maxiter_trwopt);
    toc
%     imshow(uint8(X))  
subplot(1,3,1)
imshow(uint8(B))
subplot(1,3,2)
imshow(uint8(B_Miss))
subplot(1,3,3)
imshow(uint8(X))
PSNR(X,B)
SSIM(X,B)
%%%%%%%%%%%%%%TRALS
% 
% figure()
% imshow(uint8(Data_Missing));
% saveas(gcf, strcat(filename,['/Einstein_ObservingRate' num2str(ObserveRatio) '.pdf']));
%% Complete Missing Data By TR
% TR Parameter
Omega=Q;
Data=B;
Data_Size = size(Data);
Data_Missing = reshape(T2V(Data).* T2V(Omega), Data_Size);
figure()
imshow(uint8(Data_Missing));
Data_Size = size(B);
Reshape_Dim   = [4,4,16,4,4,16,3];
r = 10;
para_TR.max_tot = 10^-4;   
para_TR.max_iter= 10;     
para_TR.disp = 1;
para_TR.r = ones(length(Reshape_Dim), 1)*r;   % Tensor Ring Rank
%para_TR.r(end)=1; % if TT

tic
% TR Completion
Utr = Completion_TR(reshape(Data_Missing, Reshape_Dim), reshape(Omega, Reshape_Dim), para_TR);
timespent=toc
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
figure()
imshow(uint8(Data_Recover_TR));
% saveas(gcf, strcat(filename,['/Einstein_Completion(TR)Rank' num2str(r) 'Err' num2str(norm(T2V(Data_Recover_TR-Data))/norm(T2V(Data))) '.pdf']));
% imwrite(uint8(Data_Recover_TR),'barbara_TR.png');
% imwrite(uint8(Data_Missing),'lena_missing1180.png')

RSE = norm(Data(:) - Data_Recover_TR(:))/norm(Data(:)); 
PSNR= PSNR(Data,Data_Recover_TR)
SSIM= SSIM(Data,Data_Recover_TR)
img_set= Data_Recover_TR;

subplot(1,3,1)
imshow(uint8(Data))
subplot(1,3,2)
imshow(uint8(Data_Missing))
subplot(1,3,3)
imshow(uint8(Data_Recover_TR))

%%%%%%%%%%%%%%%%%%%%%%%%%%SPC
clc;clear all
B =double(imread('lena.bmp')); 
% mr=0.5;
% S = size(X); 
% Q = gen_W(S,mr);
% Q=ones(256,256,3);
B_Miss=Q.*B;
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

X0 =  double(B).*Q;
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
maxiter = 150;        % maximum number of iteration
tol     = 1e-5;        % tolerance
out_im  = 1;           % you can monitor the process of 'image' completion if out == 1. 'saved' directory is necessary to save the individual rank images.

tic
[T Z G U histo histo_R] = SPC(T,Q,TVQV,rho,K,SNR,nu,maxiter,tol,out_im);
toc
B =double(imread('lena.bmp')); 
% PSNR(X,B)
% SSIM(X,B)
figure(2)
subplot(1,3,1)
imshow(uint8(B))
subplot(1,3,2)
imshow(uint8(B_Miss))
subplot(1,3,3)
imshow(uint8(T))
clear PSNR SSIM
PSNR(T,B)
SSIM(T,B)