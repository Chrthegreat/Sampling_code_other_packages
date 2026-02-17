% List of Netlib polytopes from the folder
polytopes = {'afiro', 'blend', 'beaconfd', 'scorpion', 'agg', ...
             'etamacro', 'sierra', 'degen2', 'degen3', '25fv47', ...
             '80bau3b', 'truss'};

results = struct();

for i = 1:length(polytopes)
    name = polytopes{i};
    fprintf('\n--- Processing: %s (%d/%d) ---\n', name, i, length(polytopes));
    
    try

        data = load(fullfile('coverage', 'problems', 'netlib', [name, '.mat']));
        P = data.problem;
        
        out = sample(P, 800);
  
        fprintf('Successfully sampled %s\n', name);
        
    catch ME
        fprintf('Error processing %s: %s\n', name, ME.message);
    end
end

fprintf('\nAll done!\n');