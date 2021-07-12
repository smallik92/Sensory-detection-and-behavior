
%% clear command window, workspace and close all windows, set random seed
clc; clear all; close all;
rng(1)

%% Specifications for time axis

time_obj.dt = 0.01;
time_obj.t_init = 1;
time_obj.t_on = 2;
time_obj.t_off = 5; 
time_obj.T_end = 10;
time_obj.T_end_exp = 10;

%% Specifications for latent state representation 

num = 2; %dimensionality of latent space

z1 = [1; 0]; % 'blue' stimulus
z2 = [0; 1]; % 'red' stimulus

dt = time_obj.dt; t_init = time_obj.t_init; t_on = time_obj.t_on;  
t_off = time_obj.t_off; T_end = time_obj.T_end; T_end_exp = time_obj.T_end_exp;

t_index = 0:dt:T_end - dt;

z = zeros(num, 2*length(t_index));
z(:,t_init/dt+1:(t_init+t_on)/dt) = repmat(z1, 1, t_on/dt);
% z(:, (T_end+t_init)/dt+1: (T_end+t_init+t_on)/dt) = repmat(z2, 1, t_on/dt);

%% Specifications for Neural Units

control_Num = 41; %number of neural units
overlap_Num = 11; %number of untuned neural units

%% Specifications for setting up LQR

model_params_obj.drift = -0.25;
model_params_obj.latent_penalty = 10;
model_params_obj.energy_penalty = 1;
model_params_obj.deriv_penalty = 0.1;
model_params_obj.noise_var_x = 0.1;
model_params_obj.noise_var_nu =  0;

[t_index, x_Response, nu_Response, feedback_mat] = normative_olfaction(time_obj, num, control_Num, overlap_Num,...
                                                                    model_params_obj, z);
                                                                
%% Plot Figures

%calculate the number of neural units in each category 
stim_A_num = ceil((overlap_Num + control_Num)/ 2);
stim_B_num = stim_A_num;
pure_stim_A = stim_A_num - overlap_Num;

figure, plot(t_index, x_Response(1:pure_stim_A, :), 'b');
hold on
plot(t_index, x_Response(pure_stim_A+1:control_Num-pure_stim_A,:), 'm');
plot(t_index, x_Response(control_Num - pure_stim_A+1:control_Num, :), 'r');
xlabel('Time in seconds'); ylabel('Neural activity');

%% Create PCA plots

[V, ~, ~] = simple_PCA(x_Response');
w1 = V(:,1); w2 = V(:,2); w3= V(:,3);

x_Response_pc1 = x_Response'*w1;
x_Response_pc2 = x_Response'*w2;
x_Response_pc3 = x_Response'*w3;

dur = t_init + t_on; 
figure,
plot3(x_Response_pc1(1:dur/dt-1),x_Response_pc2(1:dur/dt-1),x_Response_pc3(1:dur/dt-1),'b','LineWidth',2);
hold on
plot3(x_Response_pc1(1.9/dt),x_Response_pc2(1.9/dt),x_Response_pc3(1.9/dt), 'o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
plot3(x_Response_pc1(dur/dt),x_Response_pc2(dur/dt),x_Response_pc3(dur/dt), 'o', 'LineWidth', 2, 'MarkerFaceColor', 'k');
plot3(x_Response_pc1(dur/dt+1:T_end/dt),x_Response_pc2(dur/dt+1:T_end/dt),x_Response_pc3(dur/dt+1:T_end/dt),'b','LineWidth',1);       
grid on        
