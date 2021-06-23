function [x_t] = solve_l1magic(y,D)

x0=D'*y; %initial guess

x_t = l1eq_pd(x0, D, [], y, 1e-3);

end

