function [X] = solveGPSR_BCm(y,A)

tau = 0.01;
X = GPSR_BCBm(y, A, tau, size(A,2),0);
     
end