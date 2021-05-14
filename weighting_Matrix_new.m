function [ b_Matrix ] = weighting_Matrix_new( overlap_Num, control_Num )
 

% Let us denote the two alternative decisions as A and B respectively

% the stimulus sensitive to A and B are denoted as stim_A and stim_B


% No. of neurons sensitive to A and B

stim_A_num = ceil((overlap_Num + control_Num)/ 2);
stim_B_num = stim_A_num;

% Get the indices

stim_A_Indices = 1  : stim_A_num;
stim_B_Indices = control_Num - stim_A_num + 1 : control_Num;

% The neurons might have an overlap given by overlap_Num

pure_stim_A = stim_A_num - overlap_Num;

% Get the indices of the pure and overlapped neurons

pure_stim_A_indices = 1 : pure_stim_A;
pure_stim_B_Indices = control_Num - pure_stim_A + 1 : control_Num;
mixed_stim_indices = pure_stim_A + 1 : pure_stim_B_Indices -1 ;


stim_A_mean =  ceil(stim_A_num/2); std_A =10;
% stim_A_Weights = normpdf(1:control_Num, stim_A_mean, std_A);
stim_A_Weights = pdf('Normal',1:control_Num,stim_A_mean,std_A);
% stim_A_Weights = normpdf(1:control_Num, stim_A_mean, std_A);
stim_B_mean =  control_Num - stim_A_mean + 1; std_B =10;
%  stim_B_Weights = normpdf(1:control_Num, stim_B_Mean, std_B);
  stim_B_Weights = pdf('Normal',1:control_Num,stim_B_mean,std_B);
%  stim_B_Weights = zeros(1,control_Num);
% stim_B_Weights = normpdf(1:stim_A_num, stim_B_Mean, std_B);
% vacant_stim_A = zeros(1, control_Num - stim_A_num); 
% vacant_stim_B = zeros(1, control_Num - stim_B_num);

% Weights on the stimulus respresentation

% b_stim_A = [stim_A_Weights vacant_stim_A];
b_stim_A = stim_A_Weights;
% b_stim_B =[vacant_stim_B stim_B_Weights];
b_stim_B = stim_B_Weights;
% stim_B_Weights_mod = [stim_B_Weights(1,1:overlap_Num),zeros(1,stim_B_num-overlap_Num)];
% b_stim_B =[vacant_stim_B stim_B_Weights_mod]; 

b_Matrix_strength =10;

b_Matrix = b_Matrix_strength * [b_stim_A ; b_stim_B];

%imagesc(b_Matrix); colormap hot

%figure,plot(b_Matrix')
% 
% close(gcf)

end

