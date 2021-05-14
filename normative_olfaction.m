function [t_index, x_Response, nu_Response, feedback_mat] = normative_olfaction(time_obj, num, control_Num, overlap_Num,...
                                                                    model_params_obj, z)
 
%% Unpack the temporal details from struct                                                                
dt = time_obj.dt; 
t_init = time_obj.t_init; 
t_on = time_obj.t_on;
T_end = time_obj.T_end;
t_index = 0:dt:T_end - dt;

%% Unpack model parameters 

drift = model_params_obj.drift;
A = drift*eye(num); %State dynamics

if control_Num <= overlap_Num
    error('No. of Control Inputs less than Overlapping Control')       
end
b_Matrix = weighting_Matrix_new(overlap_Num, control_Num); %Tuning matrix

%% Dynamics for augmented state variable 

% Augmenting activity of neural units
A_modified = [A b_Matrix ;zeros(control_Num, num) zeros(control_Num)];

% Augmenting nominal representation 
aux_eig_A = -1e-10 * ones(1, num ); 
aux_A_matrix = diag(aux_eig_A);  

% Final A matrix
A_final = blkdiag(A_modified , aux_A_matrix);

% Final B matrix 
B_final = [zeros(num, control_Num); eye(control_Num); zeros(num, control_Num)];

%% Penalty matrices for LQR format 

latent_state_penalty = model_params_obj.latent_penalty; 
energy_penalty = model_params_obj.energy_penalty; 
deriv_penalty = model_params_obj.deriv_penalty; 

Q = latent_state_penalty*eye(num);
S = energy_penalty*eye(control_Num);
R = deriv_penalty*eye(control_Num); 

Q_final = [Q, zeros(num, control_Num), -Q; 
           zeros(control_Num,num), S, zeros(control_Num,num);
           -Q, zeros(num,control_Num), Q];
R_final = R; 

%% Solve the LQR problem

[W, K, e] = lqr(A_final,B_final,Q_final,R_final);
feedback_mat = W;

W_v = -W(:,1:num);
W_x = -W(:,num+1:num+control_Num); %fast connections
W_z = -W(:,num+control_Num+1:end);
W_s = W_v*b_Matrix; % slow connections

%% Forward simulation of the sensory network and latent representation 

noise_var_x = model_params_obj.noise_var_x; noise_var_nu = model_params_obj.noise_var_nu; 

% initialize response vectors 
x_Response = zeros(control_Num, length(t_index));
nu_Response = zeros(num, length(t_index)); 

for i = 1: length(t_index)-1
    nu_Response(:,i+1) = nu_Response(:,i)+dt*(drift*nu_Response(:,i)+b_Matrix*x_Response(:,i)+noise_var_nu*randn(num,1));
    x_Response(:,i+1) = x_Response(:,i)+dt*(W_v*nu_Response(:,i)+ W_x*x_Response(:,i)+W_z*z(:,i)+noise_var_x*randn(control_Num,1));
end
 
end