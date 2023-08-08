% Copilot, write me a hello world program in Matlab.

disp 'Starting experiment...'
addpath /pscratch/sd/v/vbharadw/tensor_toolbox/
addpath /pscratch/sd/v/vbharadw/tensors/

load uber;
X = uber;
clear uber;

R = 25;
M = cp_arls_lev(X, R, 'nsamplsq', 2^16);