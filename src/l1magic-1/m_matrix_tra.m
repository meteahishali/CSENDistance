function x0 = m_matrix_tra(y,m,N)

x0=zeros(N,1);
for i=1:m
    rng(i)
    a=randn(N,1);
    x0= x0 + a*y(i);
end

rng('shuffle')

end
