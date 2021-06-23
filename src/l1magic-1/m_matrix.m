function y = m_matrix(x,m,N)
y=zeros(m,1);
if size(x,2)~=1
    x=x';
    flg=1;
    
else
    flg=0;
end


for i=1:m
    rng(i)
    a=randn(N,1);
    y(i)=a'*x;
end

if flg==1
    y=y';
end

rng('shuffle')



end

