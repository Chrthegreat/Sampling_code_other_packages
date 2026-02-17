%% Example 5: Read a polytope according to Cobra format
%%initSampler
%%load(fullfile('coverage','Recon3.mat'))
%%P = struct; % Warning: Other Cobra models may have optional constraints (C,d)
%%P.lb = model.lb;
%%P.ub = model.ub;
%%P.beq = model.b;
%%P.Aeq = model.S;
%%o = sample(P, 1000);

clear; clc;
initSampler;

fprintf('Starting Sampling...\n');
data = load(fullfile('coverage','Recon1.mat'));
P = data.problem;
o = sample(P, 800);



%disp(fieldnames(data))
%disp(fieldnames(P))