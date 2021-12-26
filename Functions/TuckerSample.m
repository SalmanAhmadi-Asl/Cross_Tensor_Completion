function [Y]=TuckerSample(X,R1,R2,R3)

%C1=tenmat(X_Noisy,[1]);
C{1}=double(tenmat(X,[1]));
%C{1}=double(C1);
[S11,S12]=size(C{1});
 
C{2}=double(tenmat(X,[2]));
%C{2}=double(C2);
[S21,S22]=size(C{2});
 
C{3}=double(tenmat(X,[3]));
%C{3}=double(C3);
[S31,S32]=size(C{3});
r = randi([1 S12],1,R1);
%r=randsample(S12,R1);
D{1}=C{1}(:,r);
QQ{1}=pinv(D{1});

r = randi([1 S22],1,R2);
%r=randsample(S22,R2);
D{2}=C{2}(:,r);
QQ{2}=pinv(D{2});

r = randi([1 S32],1,R3);
%r=randsample(S32,R3);
D{3}=C{3}(:,r);
QQ{3}=pinv(D{3});

%S2=((ttensor(tensor(X),pinv(D{1}),pinv(D{2}),pinv(D{3}))));
S2=tmprod(X,QQ,[1,2,3]);
%S2=lmlragen(Q,double(X_Noisy));
Y=tmprod(S2,D,[1,2,3]);
end