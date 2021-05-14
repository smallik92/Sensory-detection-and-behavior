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

control_Num = 41; %number of neural units
overlap_Num = 11; %number of untuned neural units

%% Specifications for setting up LQR

model_params_obj.drift = -0.25;
model_params_obj.latent_penalty = 10;
model_params_obj.energy_penalty = 2;
model_params_obj.deriv_penalty = 0.1;
model_params_obj.noise_var_x = 0.025;
model_params_obj.noise_var_nu =  0;

[~, ~, ~, feedback_mat] = normative_olfaction(time_obj, num, control_Num, overlap_Num,...
                                                                    model_params_obj, z);
                                                                
%% 

W_v = -feedback_mat(:,1:num);
W_x = -feedback_mat(:,num+1:num+control_Num); 
W_z = -feedback_mat(:,num+control_Num+1:end); 

%% Extract required parameters

dt = time_obj.dt; t_init = time_obj.t_init; t_on = time_obj.t_on;
T_end_exp = time_obj.T_end ;
t_index = 0:dt:T_end_exp-dt;

t_mid = T_end_exp/2;

drift = model_params_obj.drift;
b_Matrix = weighting_Matrix_new(overlap_Num, control_Num);

%% Simulate and plot
nu_Response = zeros(num,length(t_index));
x_Response = zeros(control_Num,length(t_index));
nu_Response_1 = zeros(num,length(t_index));

for jj = 1:length(t_index)-1
x_Response(:,jj+1) = x_Response(:,jj)+dt*(W_v*nu_Response(:,jj)+W_x*x_Response(:,jj)+W_z*z(:,jj));
nu_Response(:,jj+1) = nu_Response(:,jj)+dt*(drift*nu_Response(:,jj)+b_Matrix*x_Response(:,jj));
end

figure, plot(nu_Response(1,1:t_mid/dt),nu_Response(2,1:t_mid/dt),'b', 'LineWidth',2);
hold on 
plot(nu_Response(1,t_mid/dt+1:end),nu_Response(2,t_mid/dt+1:end),'r', 'LineWidth',2);

x_Response_1 = x_Response;
x_Response_1(:,(t_init+t_on)/dt+1:t_mid/dt) = 0;
x_Response_1(:,(t_mid+t_init+t_on)/dt+1:end) = 0;
for jj = 1:length(t_index)-1
%x_Response_1(:,jj+1) = x_Response_1(:,jj)+dt*(W_v*nu_Response(:,jj)+W_x*x_Response_1(:,jj)+W_z*z(:,jj));
nu_Response_1(:,jj+1) = nu_Response_1(:,jj)+dt*(drift*nu_Response_1(:,jj)+b_Matrix*x_Response_1(:,jj));
end

hold on
plot(nu_Response_1(1,1:t_mid/dt),nu_Response_1(2,1:t_mid/dt),'--b','LineWidth',2);
hold on 
plot(nu_Response_1(1,t_mid/dt+1:end),nu_Response_1(2,t_mid/dt+1:end),'--r','LineWidth',2);
xlim([-0.2,1]); ylim([-0.2,1]);
grid on
ax = gca;
ax.FontSize = 18;
ax.FontName = 'Arial';