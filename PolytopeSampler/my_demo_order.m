% Experiment Parameters
dims = [500]; 
targetESS = 800;
base_seed = 42;
resultsData = zeros(length(dims), 3); % [Dim, Facets, Time]

if exist('initSampler', 'file') == 2
    initSampler;
end

fprintf('Starting Order Polytope Experiment (Target ESS: %d)...\n', targetESS);

for i = 1:length(dims)
    dim = dims(i); 
    
    % Replicate the C++ facet math: m = 3 * dim
    m = 4 * dim; 
    
    % The bounding box (0 <= x <= 1) accounts for 2*dim facets.
    % The remaining facets are the random poset relations.
    num_relations = m - 2 * dim; 
    
    fprintf('\n========================================\n');
    fprintf('Dimension: %d | Total Facets: %d | Random Relations: %d\n', ...
            dim, m, num_relations);
    fprintf('========================================\n');
    
    % --- REPLICATE VOLESTI RANDOM POSET LOGIC ---
    
    % Set seed (Mersenne Twister is the closest to C++ mt19937)
    current_seed = base_seed + dim;
    rng(current_seed, 'twister'); 
    
    % 1. Create and shuffle the order (MATLAB uses 1-based indexing)
    order_vec = randperm(dim); 
    
    % --- DEFINE ORDER POLYTOPE CONSTRAINTS ---
    P = struct;
    
    % 2. Bounding Box: 0 <= x_i <= 1
    P.lb = zeros(dim, 1);
    P.ub = ones(dim, 1);
    
    % 3. Inequality Constraints: Aineq*x <= bineq
    P.Aineq = zeros(num_relations, dim);
    P.bineq = zeros(num_relations, 1);
    
    for k = 1:num_relations
        % Sample x and y uniformly
        x = randi([1, dim]);
        y = randi([1, dim]);
        
        % Ensure x != y
        while x == y
            y = randi([1, dim]);
        end
        
        % Ensure x < y (swap if greater)
        if x > y
            temp = x;
            x = y;
            y = temp;
        end
        
        % The relation is: order_vec(x) <= order_vec(y)
        % In matrix form: 1*u - 1*v <= 0
        u = order_vec(x);
        v = order_vec(y);
        
        P.Aineq(k, u) = 1;
        P.Aineq(k, v) = -1;
        P.bineq(k) = 0;
    end
    
    % --- SAMPLER OPTIONS ---
    opts = default_options();

    % --- SAMPLE ---
    tic;
    o = sample(P, targetESS, opts); 
    rawTime = toc; 
    
    % Store: [Dim, Facets, Time]
    resultsData(i, :) = [dim, m, rawTime];


    % --- PLOT 3D SAMPLES ---
    if dim == 3
        % NOTE: Depending on your specific MATLAB package, 'o' might be the 
        % matrix of samples, or it might be a struct (like o.samples). 
        % Adjust this next line if necessary!
        samples3D = o.samples; 
        
        % Make sure the matrix is N x 3 (Rows = points, Columns = coordinates)
        if size(samples3D, 2) ~= 3 && size(samples3D, 1) == 3
            samples3D = samples3D'; 
        end
        
        figure('Name', '3D Order Polytope Samples', 'Position', [200, 200, 600, 600]);
        
        % Plot the points using a semi-transparent blue scatter plot
        scatter3(samples3D(:,1), samples3D(:,2), samples3D(:,3), 15, 'filled', ...
            'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', [0 0.4470 0.7410]);
        
        % Force the axes to show the strict [0, 1] bounding box of the order polytope
        xlim([0, 1]);
        ylim([0, 1]);
        zlim([0, 1]);
        
        xlabel('X_1 (Coordinate 1)', 'FontWeight', 'bold');
        ylabel('X_2 (Coordinate 2)', 'FontWeight', 'bold');
        zlabel('X_3 (Coordinate 3)', 'FontWeight', 'bold');
        title(sprintf('Order Polytope Samples (Dim = 3, Relations = %d)', num_relations));
        
        grid on;
        view(45, 30); % Set a nice isometric viewing angle
        drawnow;      % Force MATLAB to draw the plot immediately
    end





end

% ---- Display Final Results Table ----
fprintf('\n\nFinal Results Table:\n');
T = array2table(resultsData, 'VariableNames', ...
    {'Dimension', 'TotalFacets', 'RawTime_s'});
disp(T);