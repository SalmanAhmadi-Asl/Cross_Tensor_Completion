function [Y]=FSTDsmooth(X,R1,R2,R3,smoothmethod,span)

S_X=size(X);
C1=tenmat(X,[1]);
C2=tenmat(X,[2]);
C3=tenmat(X,[3]);

%r1 = randi([1 50],1,t1);
r1=randsample(S_X(1),R1);

%r2 = randi([1 66],1,t2);
r2=randsample(S_X(2),R2);

%r3 = randi([1 45],1,t3);
r3=randsample(S_X(3),R3);

W=X(r1,r2,r3);
W1=tenmat(W,[1]);
W2=tenmat(W,[2]);
W3=tenmat(W,[3]);

B1=double(tenmat(X(:,r2,r3),[1]));

SZ_B1=size(B1,2);

for i=1:SZ_B1
    switch smoothmethod 
        case {'moving' 'lowess' 'sgolay' 'rlowess'}
            BB1(:,i)=smooth(B1(:,i),span,smoothmethod);
        case 'wavelet'
            BB1(:,i) = wdenoise(B1(:,i),3, ... %floor(log2(size(D{1}(:,i),1)))
                'Wavelet', 'db4', ...
                'DenoisingMethod', 'Bayes', ...
                'ThresholdRule', 'Median', ...
                'NoiseEstimate', 'LevelIndependent');
    end
  % D{1}(:,i)=smooth(Wf{1}(:,r(i)),D{1}(:,i));
end

% for i=1:SZ_B1
%     BB1(:,i)=smooth(B1(:,i));
% end

B2=double(tenmat(X(r1,:,r3),[2]));


SZ_B2=size(B2,2);

for i=1:SZ_B2
    switch smoothmethod 
        case {'moving' 'lowess' 'sgolay' 'rlowess'}
            BB2(:,i)=smooth(B2(:,i),span,smoothmethod);
        case 'wavelet'
            BB2(:,i) = wdenoise(B2(:,i),3, ... %floor(log2(size(D{1}(:,i),1)))
                'Wavelet', 'db4', ...
                'DenoisingMethod', 'Bayes', ...
                'ThresholdRule', 'Median', ...
                'NoiseEstimate', 'LevelIndependent');
    end
  % D{1}(:,i)=smooth(Wf{1}(:,r(i)),D{1}(:,i));
end

% for i=1:SZ_B2
%     BB2(:,i)=smooth(B2(:,i));
% end

B3=double(tenmat(X(r1,r2,:),[3]));

SZ_B3=size(B3,2);

% for i=1:SZ_B3
%     BB3(:,i)=smooth(B3(:,i));
% end

Y=full(ttensor(tensor(W),double(BB1)*pinv(double(W1)),double(BB2)*pinv(double(W2)),double(B3)*pinv(double(W3))));
end