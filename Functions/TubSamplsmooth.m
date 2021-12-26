function [Y]=TubSamplsmooth(X,R1,R2)
N_m=size(X);
FX=fft(X,[],3);

r1=randsample(N_m(2),R1);
FC=X(:,r1,:);
%FC=reshape(smooth(FC),[N_m(1),size(r1,1),N_m(3)]);
SZ_r1=size(r1);
for i=1:SZ_r1
    for j=1:N_m(3)
FC(:,i,j)=smooth(squeeze(FC(:,i,j)),3);        
    end
end
% FC=fft(FC,[],3);

r2=randsample(N_m(1),R2);
FR=X(r2,:,:);
% SZ_r2=size(r2);
% for i=1:SZ_r2
%     for j=1:N_m(3)
% FR(i,:,j)=smooth(squeeze(FR(i,:,j)));        
%     end
% end
%FR=reshape(smooth(FR),[size(r2,1),N_m(2),N_m(3)]);
% FC=fft(FR,[],3);

% for i=1:
%    Y1(:,:,) 
% end
U=tprod(tprod(t_pinv(FC),X),t_pinv(FR));
% U=t_pinv(X(r2,r1,:));

Y=tprod(tprod(FC,U),FR);
end