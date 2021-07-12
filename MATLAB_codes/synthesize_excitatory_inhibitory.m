function [W_hat_final, W_bar_final] = synthesize_excitatory_inhibitory(maxIter, n_o, n_i, lambda_1, lambda_2, W_s)

if nargin<6
    error('Provide the slow connectivity matrix...');
end

if nargin<5
    lambda_2 = 0.3;
end

if nargin<4
   lambda_1 = 0.1;
end

   
%%
W_bar = zeros(n_i,n_o); %initialize

% Change here for changing initialization
%Clustered random 
tune_A_num = floor(0.5*n_i);
stim_A_num = floor(0.5*n_o);

partial_mat_upper = 0.1*rand(tune_A_num, stim_A_num);
W_bar(1:tune_A_num, 1:stim_A_num) = partial_mat_upper;

partial_mat_lower = 0.1*rand(tune_A_num,stim_A_num);
W_bar(n_i-tune_A_num+1:end, n_o - stim_A_num+1:end) = W_bar(n_i-tune_A_num+1:end, n_o - stim_A_num+1:end) + partial_mat_lower;

 %% Convex Optimization to deduce W_hat and W_bar
 

best_W_bar = W_bar;
best_W_hat = pinv(W_bar);

flag = 0; %internal communication to signal presence of a solution

obj_curr = zeros(1, maxIter);
best_error = zeros(1,maxIter);

best_error(1) = norm(pinv(W_bar)*W_bar - eye(n_o), 'fro');
obj_curr(1) = NaN;

for i = 2:maxIter
    cvx_begin quiet
        variable W_hat(n_o,n_i) 
        minimize(norm(W_hat*W_bar - eye(n_o), 'fro') + lambda_1*norm(W_hat,'fro'))
        Z = W_s*W_hat;
        Z(:) <= 0;
    cvx_end

    cvx_begin quiet
        variable W_bar(n_i,n_o)
        minimize (norm(W_hat*W_bar - eye(n_o),'fro') + lambda_2*norm(W_bar,'fro'))
        W_bar(:) >= 0;
    cvx_end

    obj_curr(i) = norm(W_hat*W_bar - eye(n_o), 'fro');

    if (obj_curr(i) <= best_error(i-1))
        best_error(i) = obj_curr(i);
        best_W_bar = W_bar;
        best_W_hat = W_hat;
        flag = 1;
    else
        best_error(i) = best_error(i-1);
    end
end

if flag == 1
    W_bar_final = best_W_bar; W_hat_final = best_W_hat;
else
    error('No solution found, try re-initializing')
end
    


figure,
plot(1:maxIter, obj_curr, 'LineWidth', 2);
hold on
plot(1:maxIter, best_error, 'LineWidth', 2);

xlabel(' Iteration #'); 
legend('Objective function', 'Best reconstruction error');
