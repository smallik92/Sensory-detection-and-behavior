%% Clear workspace

% Note: You can create PCA trajectories for comparing between on response
% in presence of two different stimuli using the same setup. Synthesize the
% neural activity appropriately and use flag = 2 for comparing on response
% (blue) vs on response(red) and flag = 3 for comparing off response (blue)
% vs off response(red).

clc;
clear all;
close all; 

%% Run code to generate neural activity

normative_olfaction_starter_demo;

%% Create PCA trajectories in three dimension for ON vs OFF comparison

flag = 1; %for comparing on vs off
x_first_trial = x_Response(:, 1:T_end/dt);
[x_Response_pc1,x_Response_pc2,x_Response_pc3] = generate_PCA_trajectories(x_first_trial, flag, dt, t_init, t_on, t_off);

figure, 
imagesc(corrcoef(x_first_trial)); colormap gray; colorbar;

%% Create PCA trajectories in three dimension for ON(blue) vs ON(red)

flag = 3; 
[x_Response_pc1_v2,x_Response_pc2_v2,x_Response_pc3_v2] = generate_PCA_trajectories(x_Response, flag, dt, t_init, t_on, t_off);