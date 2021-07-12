clc; clear; close all;
rng(2)

%% Specifications for time axis

time_obj.dt = 0.01;
time_obj.t_init = 0;
time_obj.t_on = 5;
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

z = zeros(num, length(t_index));
z(:,t_init/dt+1:(t_init+t_on)/dt) = repmat(z1, 1, t_on/dt);

%% Specifications for Neural Units

control_Num = 41; %number of neural units
overlap_Num = 11; %number of untuned neural units

%% Specifications for setting up LQR

model_params_obj.drift = -0.2;
% model_params_obj.latent_penalty = 10;
% model_params_obj.energy_penalty = 2;
model_params_obj.deriv_penalty = 0.1;
model_params_obj.noise_var_x = 0;
model_params_obj.noise_var_nu =  0;

energy_penalty_array = 2:2:24; 
latent_penalty_array = 2:2:30;

error = zeros(length(energy_penalty_array), length(latent_penalty_array));
reaction_time = zeros(length(energy_penalty_array), length(latent_penalty_array));
color = linspace(0, 1, length(energy_penalty_array));

figure,
for k = 1:length(energy_penalty_array)
    model_params_obj.energy_penalty = energy_penalty_array(k);
    for l = 1:length(latent_penalty_array)
        model_params_obj.latent_penalty = latent_penalty_array(l);
        [t_index, x_Response, nu_Response, feedback_mat] = normative_olfaction(time_obj, num, control_Num, overlap_Num,...
                                                                    model_params_obj, z);
                                                                
        error(k, l) = norm(nu_Response(:, (t_init+t_on)/dt) - z1);
        
        for t = 1:t_on/dt
            if norm(nu_Response(:, t)-z1) < 0.3
                reaction_time(k, l) = t*dt;
                break
            end
        end
        
        
    end
    
    plot(0:dt:t_on - dt, mean(x_Response(1:10, 1:(t_init+t_on)/dt)), 'LineWidth', 1.0, 'Color', [0; 0; color(k)]);
    hold on
    ax = gca;
    ax.FontSize = 18;
    ax.FontName = 'Arial';
    ax.XTick = [0,1, 2,3,4,5];
end

%%

accuracy = 1-error;
[X,Y] = meshgrid(latent_penalty_array, energy_penalty_array);

figure, surf(X, Y, accuracy); colormap pink; 
xlabel('Latent Error Penalty');
ylabel('Energy Penalty');
zlabel('Accuracy');

figure, surf(X, Y, reaction_time); colormap pink; 
xlabel('Latent Error Penalty');
ylabel('Energy Penalty');
zlabel('Latency');