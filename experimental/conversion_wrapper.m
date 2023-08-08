filename = '/pscratch/sd/v/vbharadw/tensors/amazon-reviews.tns'; 

addpath /pscratch/sd/v/vbharadw/tensor_toolbox/
rawdata = readmatrix(filename, 'Filetype', 'text');

%tensor = sptensor(rawdata(:,1:3), rawdata(:,4));
tensor = sptensor(rawdata(:,1:3), log(rawdata(:,4)) + 1);

R = 25;
%M = cp_arls_lev(tensor, R, 'nsamplsq', 2^16, 'maxepochs', 1);
tic; M = cp_als(tensor, R, 'maxiters', 1, 'printitn', 0); elapsed = toc;
fprintf('ALS took %f seconds\n', elapsed);

exit;