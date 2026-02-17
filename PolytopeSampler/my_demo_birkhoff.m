% Experiment Parameters
% dims represents the side length 'n' of the matrix.
% Intrinsic dimension will be (n-1)^2.
dims = [5 10 15 20 25 30 35 40 45 50 55]; 

targetESS = 800;
resultsData = zeros(length(dims), 3); % [n, IntrinsicDim, Time]

if exist('initSampler', 'file') == 2
    initSampler;
end

fprintf('Starting Birkhoff Polytope Experiment (Target ESS: %d)...\n', targetESS);

for i = 1:length(dims)
    n = dims(i); 
    
    % Ambient variables (entries in matrix)
    n_vars = n^2; 
    
    % Intrinsic dimension (degrees of freedom)
    dim_intrinsic = (n-1)^2;
    
    fprintf('\n========================================\n');
    fprintf('Matrix Size: %dx%d | Ambient Vars: %d | Intrinsic Dim: %d\n', ...
            n, n, n_vars, dim_intrinsic);
    fprintf('========================================\n');
    
    % Define Birkhoff in the same way as in the provided demo
    P = struct;
    P.lb = zeros(n_vars, 1);
    P.Aeq = sparse(2*n, n_vars);
    P.beq = ones(2*n, 1);
    
    for k = 1:n
        % Row k sum
        P.Aeq(k, (k-1)*n+1 : k*n) = 1;
        % Col k sum
        P.Aeq(n+k, k : n : n_vars) = 1;
    end
    
    % SAMPLER OPTIONS 
    opts = default_options();
    % Start at the center (matrix of all 1/n)
    opts.x0 = (1/n) * ones(n_vars, 1); 
    
    % Sample using the sample function
    tic;
    o = sample(P, targetESS, opts); 
    rawTime = toc; 
    
    % Store: [n, IntrinsicDim, Time]
    resultsData(i, :) = [n, dim_intrinsic, rawTime];
end

fprintf('\n\nFinal Results Table:\n');
T = array2table(resultsData, 'VariableNames', ...
    {'MatrixSide_n', 'IntrinsicDim', 'RawTime_s'});
disp(T);



