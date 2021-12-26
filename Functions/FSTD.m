function [Y]=FSTD(X,R1,R2,R3)

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

B2=double(tenmat(X(r1,:,r3),[2]));

B3=double(tenmat(X(r1,r2,:),[3]));

Y=full(ttensor(tensor(W),double(B1)*pinv(double(W1)),double(B2)*pinv(double(W2)),double(B3)*pinv(double(W3))));
end