
%% Clear workspace
clc; 
clear all;
rng(2)

%% Specifications for simulation

dt =0.01; %interval of integration
t_init = 1; %in seconds
t_on = 4; %in seconds
t_off = 5; %in seconds
T_end = t_init+t_on+t_off;
t_index =0:dt:T_end-dt;


latent_dim = 2;  z1 = [1;0];     % Dimension of feature space
z = zeros(latent_dim, T_end/dt); % stimulus vector
z(:,t_init/dt+1:(t_init+t_on)/dt)=repmat(z1,1,t_on/dt);

control_Num = 41;                  % No. of PNs
overlap_Num =11;                   % No of overlap
inner_Num = ceil(0.75*control_Num);        % No. of LNs

b_Matrix  = weighting_Matrix_new( overlap_Num, control_Num );

%System dynamics
drift = -0.25;
A = [drift 0; 0 drift]; 

% Modification to A matrix for augmenting control as a state variable
A_modified = [A b_Matrix ;zeros(control_Num, latent_dim) zeros(control_Num)];
 
% Modification to A matrix for the augmented variables for threshold tracking         
aux_eig_A = -1e-10 * ones(1, latent_dim ); 
aux_A_matrix = diag(aux_eig_A); 
A_final = blkdiag(A_modified , aux_A_matrix);

b_modified = [zeros(latent_dim , control_Num); eye(control_Num)];
aux_b = zeros(latent_dim, control_Num);
B_final = [b_modified ; aux_b];

% Penalty matrices
target_Param = 10; energy_Param = 2; deriv_Param = .1;

Q = diag([target_Param target_Param]);
R_control = energy_Param * eye(control_Num);
Q_final = [Q zeros(latent_dim, control_Num) -Q ; ...
            zeros(control_Num, latent_dim) R_control zeros(control_Num, latent_dim);...
            -Q  zeros(latent_dim, control_Num) Q];
        
R_control_deriv = deriv_Param * eye(control_Num);  

[K, P, e] = lqr(A_final,B_final,Q_final,R_control_deriv);

W_v = - K(:,1:latent_dim);
W_x = - K(:,latent_dim+1:control_Num+latent_dim);
W_z = - K(:,control_Num+latent_dim+1:end);

% Deducing synaptic connections

W_f = W_x; %fast synapses
W_s = W_v*b_Matrix; %slow synapses

%% Network decomposition algorithm 

maxIter = 20; lambda_1 = 0.1; lambda_2 = 0.3;
[W_hat, W_bar] = synthesize_excitatory_inhibitory(maxIter, control_Num, inner_Num, lambda_1, lambda_2, W_s);

figure,
subplot(2,2,1); imagesc(drift*eye(inner_Num)); colormap gray; colorbar
subplot(2,2,2); imagesc(W_bar); colormap gray; colorbar
subplot(2,2,3); imagesc(W_s*W_hat); colormap gray; colorbar;
subplot(2,2,4); imagesc(W_f); colormap gray; colorbar;

%% Spatiotemporal activity

x_Response_o = zeros(control_Num, length(t_index));
x_Response_i = zeros(inner_Num, length(t_index));

for t= 2:length(t_index)
    x_Response_i(:,t) =x_Response_i(:,t-1)+dt*(drift*x_Response_i(:,t-1)+ W_bar*x_Response_o(:,t-1)+0.05*randn);
    x_Response_o(:,t) = x_Response_o(:,t-1)+dt*(W_s*W_hat*x_Response_i(:,t-1)+ W_f*x_Response_o(:,t-1)+W_z*z(:,t-1)+0.05*randn);
end

figure, 
plot(t_index, x_Response_o); 
xlabel('Time')
ylabel(' Representative PN activity')


figure,
plot(t_index, x_Response_i);
xlabel('Time');
ylabel(' Representative LN activity') ;