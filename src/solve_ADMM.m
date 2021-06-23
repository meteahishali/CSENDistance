function [aa] = solve_ADMM(y,D)

xp_l2 = pinv (D)*y ; % minimum norm solution
x_o=xp_l2;
params.maxit=50;
params.lambda=0.01;
params.mu=0.8;
params.model='l1';
%0.1*0.08
params.N=size(D,2);%  n;

%[aa,residual_norm, x_norm] = ADMM_lasso(y,D,params,xp_l2,x_o);
lambda_max = norm( D'*y, 'inf' );
lambda = 0.1*lambda_max;

[aa ~] = lasso(D, y, lambda, 1.0, 1.0);

%aa=SolveOMP(D, y,'tolerance',0.01);
end