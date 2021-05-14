%% clear command window, workspace and close all windows, set random seed
clc; clear all; close all;
rng(1)

%% Specifications for time axis

time_obj.dt = 0.01;
time_obj.t_init = 1;
time_obj.t_on = 4;
time_obj.t_off = 5; 
time_obj.T_end = 10;

%% Specifications for latent state representation 

num = 2; %dimensionality of latent space

z1 = [1; 0]; % 'blue' stimulus
z2 = [0; 1]; % 'red' stimulus

dt = time_obj.dt; t_init = time_obj.t_init; t_on = time_obj.t_on; T_end = time_obj.T_end;

t_index = 0:dt:T_end-dt;

z = zeros(num, length(t_index));
z(:,t_init/dt+1:(t_init+t_on)/dt) = repmat(z1, 1, t_on/dt);

%% Specifications for Neural Units

control_Num = 41; %number of neural units
overlap_Num = 11; %number of untuned neural units

%% Specifications for setting up LQR

model_params_obj.drift = -0.25;
model_params_obj.latent_penalty = 10;
model_params_obj.energy_penalty = 2;
model_params_obj.deriv_penalty = 0.1;
model_params_obj.noise_var_x = 0.025;
model_params_obj.noise_var_nu =  0;

[t_index, x_Response, nu_Response, feedback_mat] = normative_olfaction(time_obj, num, control_Num, overlap_Num,...
                                                                    model_params_obj, z);
                                                                
%% Plot Figures

%calculate the number of neural units in each category 
stim_A_num = ceil((overlap_Num + control_Num)/ 2);
stim_B_num = stim_A_num;
pure_stim_A = stim_A_num - overlap_Num;

figure, plotshaded(t_index, x_Response(1:pure_stim_A, :), 'b');
hold on
plotshaded(t_index, x_Response(pure_stim_A+1:control_Num-pure_stim_A,:), 'm');
plotshaded(t_index, x_Response(control_Num - pure_stim_A+1:control_Num, :), 'r');
xlabel('Time in seconds'); ylabel('Neural activity');

figure, plot(nu_Response(1,:), nu_Response(2,:), 'LineWidth', 1.5, 'Color', 'b');
axis([-0.2, 1.05, -0.2, 1.05]); grid on
xlabel('\nu_1'); ylabel('\nu_2');

