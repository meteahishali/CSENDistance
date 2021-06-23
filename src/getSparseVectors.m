function X = getSparseVectors(Y,A,l1method)

% Normalize the columns of Y to have unit l^2-norm.
%for i = 1 : size(Y,2)
%    Y(:,i) = Y(:,i) ./ (norm(Y(:,i))+eps);
%end


mtest = mean(Y, 2);

mtest = mtest ./ (norm(mtest)+eps);
%tau = 0.1;
%X = GPSR_BCBm(mtest, A, tau, size(A,2),0);
method=str2func(l1method);
X=method(mtest,A);

return