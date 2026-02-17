% List of Netlib polytopes
polytopes = {'afiro', 'blend', 'beaconfd', 'scorpion', 'agg', ...
             'etamacro', 'sierra', 'degen2', 'degen3', '25fv47'};

inputDir = fullfile('coverage', 'problems', 'netlib');
outputDir = 'volesti_input';
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

% 1e-4 provides a safer buffer for the Ellipsoid method to work.
epsilon = 1e-4; 

% Reduce the infinity value for better stabillity
INF_VAL = 1e6;

for i = 1:length(polytopes)
    name = polytopes{i};
    fprintf('\n--- Processing: %s ---\n', name);
    
    try
        % Load the problem
        if ~exist(fullfile(inputDir, [name, '.mat']), 'file')
             warning('File %s.mat not found. Skipping.', name);
             continue;
        end
        data = load(fullfile(inputDir, [name, '.mat']));
        P = data.problem;
        
        Aeq = full(P.Aeq); 
        beq = full(P.beq);
        lb = P.lb;
        ub = P.ub;
        
        [m, n] = size(Aeq);
        
        % Find a Feasible x0 
        % Note: Using dual-simplex is good, but interior-point might 
        % find a more "centered" x0, keeping away from bounds.
        fprintf('  Searching for a feasible x0...\n');
        f = zeros(n, 1);
        opts = optimoptions('linprog', 'Display', 'none', 'Algorithm', 'dual-simplex');
        [x0, ~, exitflag] = linprog(f, [], [], Aeq, beq, lb, ub, opts);
        
        if exitflag ~= 1
            warning('  Could not find feasible x0 for %s. Skipping.', name);
            continue;
        end
        
        % Calculate Orthonormal Null Space
        fprintf('  Computing orthonormal null space...\n');
        Z = null(Aeq); 
        
        if isempty(Z)
            warning('  Null space is empty (unique solution). Skipping %s.', name);
            continue;
        end
        
        % Transform Bounds to H-representation with Padding
        % The inequality is: lb <= x0 + Z*y <= ub
        % Which becomes: 
        %  Z*y <= ub - x0
        % -Z*y <= x0 - lb
        
        A_volesti = [Z; -Z];
        b_volesti = [(ub - x0); (x0 - lb)];
        
        % Add epsilon padding (Fattening)
        b_volesti = b_volesti + epsilon;

        % Handle Infinities 
        b_volesti(b_volesti > INF_VAL) = INF_VAL;
        
        % Explicit Precision with writematrix 
        fprintf('  Exporting CSVs (Dim: %d)...', size(Z,2));
        
        fileA = fullfile(outputDir, [name, '_A.csv']);
        fileB = fullfile(outputDir, [name, '_b.csv']);
        
        % Write A matrix 
        writematrix(A_volesti, fileA);
        
        % Write b vector (Ensure it is a column vector)
        writematrix(b_volesti, fileB);
        
        fprintf(' Done!\n');
        
    catch ME
        fprintf('  Error on %s: %s\n', name, ME.message);
    end
end