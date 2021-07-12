%% draw PCA trajectories 
function[x_Response_pc1,x_Response_pc2,x_Response_pc3] = generate_PCA_trajectories(x, flag, dt, t_init, t_on, t_off)
[V, ~, ~] = my_pca(x');

w1 = V(:,1); w2 = V(:,2); w3= V(:,3);
T_end = t_init + t_on + t_off;

switch flag
    case 1 %on vs off
        dur = t_init + t_on;
        x_data = x; 
        x_Response_pc1 = x_data'*w1;
        x_Response_pc2 = x_data'*w2;
        x_Response_pc3 = x_data'*w3;
        
        %creates the figure
        figure,
        for i =1:10:length(x_data)-1
            if(i<=length(x_data)/2)
                plot3([0,x_Response_pc1(i)],[0,x_Response_pc2(i)],[0,x_Response_pc3(i)],'r','LineWidth',1.0);
                hold on
            else
                plot3([0,x_Response_pc1(i)],[0,x_Response_pc2(i)],[0,x_Response_pc3(i)],'b','LineWidth',1.0);
                hold on
            end
        end
       hold on
       plot3(x_Response_pc1(1:dur/dt-1),x_Response_pc2(1:dur/dt-1),x_Response_pc3(1:dur/dt-1),'r','LineWidth',1.5);
       plot3(x_Response_pc1(dur/dt+1:T_end/dt),x_Response_pc2(dur/dt+1:T_end/dt),x_Response_pc3(dur/dt+1:T_end/dt),'b','LineWidth',1.5);       
        
     case 2 %on (blue) vs on (red)
            dur = t_on; 
            x_data = zeros(size(x,1), 2*dur/dt);
            x_data(:,1:dur/dt) = x(:,t_init/dt + 1:(t_init + t_on)/dt);
            x_data(:,dur/dt+1:end) = x(:,(T_end+t_init)/dt + 1: (T_end + t_init+t_on)/dt);
            
            x_Response_pc1 = smooth(x_data'*w1);
            x_Response_pc2 = smooth(x_data'*w2);
            x_Response_pc3 = smooth(x_data'*w3);
            figure,
         for i =1:10:length(x_data)-1
            if(i<=length(x_data)/2)
                plot3([0,x_Response_pc1(i)],[0,x_Response_pc2(i)],[0,x_Response_pc3(i)],'b','LineWidth',1.5);
                  hold on
                  
            else
                 plot3([0,x_Response_pc1(i)],[0,x_Response_pc2(i)],[0,x_Response_pc3(i)],'r','LineWidth',1.5);
                  hold on
                 
            end
        end
        hold on
        plot3(x_Response_pc1(1:dur/dt-1),x_Response_pc2(1:dur/dt-1),x_Response_pc3(1:dur/dt-1),'b','LineWidth',1.5);
        plot3(x_Response_pc1(dur/dt+1:end),x_Response_pc2(dur/dt+1:end),x_Response_pc3(dur/dt+1:end),'r','LineWidth',1.5);
            
    case 3 %off (blue) vs off (red)
        dur = t_off;
        x_data = zeros(size(x,1), 2*dur/dt);
        x_data(:,1:dur/dt) = x(:,dur/dt+1:T_end/dt );
        x_data(:,dur/dt+1:end) = x(:,(T_end+dur)/dt+1: 2*T_end/dt);
        
        x_Response_pc1 = smooth(x_data'*w1);
        x_Response_pc2 = smooth(x_data'*w2);
        x_Response_pc3 = smooth(x_data'*w3);
        figure,
     for i =1:5:length(x_data)-1
        if(i<=length(x_data)/2)
            plot3([0,x_Response_pc1(i)],[0,x_Response_pc2(i)],[0,x_Response_pc3(i)],'b','LineWidth',1.5);
              hold on
        else
             plot3([0,x_Response_pc1(i)],[0,x_Response_pc2(i)],[0,x_Response_pc3(i)],'r','LineWidth',1.5);
              hold on
        end
    end
    hold on
    plot3(x_Response_pc1(1:dur/dt-2),x_Response_pc2(1:dur/dt-2),x_Response_pc3(1:dur/dt-2),'b','LineWidth',1.5);
    plot3(x_Response_pc1(dur/dt+1:end),x_Response_pc2(dur/dt+1:end),x_Response_pc3(dur/dt+1:end),'r','LineWidth',1.5);
end

grid on
ax = gca;
ax.FontSize = 18;
ax.FontName = 'Arial';
hold off
