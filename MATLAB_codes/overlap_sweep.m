%% clear command window, workspace and close all windows, set random seed
clc; clear all; close all;
rng(1)

%% Specifications for time axis

time_obj.dt = 0.01;
time_obj.t_init = 1;
time_obj.t_on = 4;
time_obj.t_off = 5; 
time_obj.T_end = 20;

%% Specifications for latent state representation 

num = 2; %dimensionality of latent space

z1 = [1; 0]; % 'blue' stimulus
z2 = [0; 1]; % 'red' stimulus

dt = time_obj.dt;
t_init = time_obj.t_init; t_on = time_obj.t_on; t_off = time_obj.t_off;
T_end = time_obj.T_end;

t_index = 0:dt:T_end-dt;

z = zeros(num, length(t_index));
z(:,t_init/dt+1:(t_init+t_on)/dt) = repmat(z1, 1, t_on/dt);
z(:,(2*t_init+t_on+t_off)/dt+1:(2*t_init+2*t_on+t_off)/dt) = repmat(z2, 1, t_on/dt);

%% Specifications for Neural Units

control_num = 41;

%% Specifications for setting up LQR

model_params_obj.drift = -0.25;
model_params_obj.latent_penalty = 10;
model_params_obj.energy_penalty = 2;
model_params_obj.deriv_penalty = 0.1;
model_params_obj.noise_var_x = 0.025;
model_params_obj.noise_var_nu =  0;

overlap_array = [1:5:40, 40];
fignum = 1;

figure,
for overlap_num = overlap_array
    [t_index, x_Response, nu_Response, feedback_mat] = normative_olfaction(time_obj, num, control_num, overlap_num,...
                                                                    model_params_obj, z);
    subplot(3,3, fignum)
    plot(nu_Response(1, 1:(t_init+t_on+t_off)/dt), nu_Response(2, 1:(t_init+t_on+t_off)/dt), 'b', 'LineWidth', 2);
    hold on
    plot(nu_Response(1, (t_init+t_on+t_off)/dt + 1:end), nu_Response(2, (t_init+t_on+t_off)/dt + 1: end), 'r', 'LineWidth', 2);
    axis([-0.3, 1.05, -0.3, 1.05]); grid on
    xlabel('\nu_1'); ylabel('\nu_2');
    fignum = fignum + 1;
end

