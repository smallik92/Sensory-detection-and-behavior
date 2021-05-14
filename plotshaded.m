function varargout = plotshaded(x,y,fstr,n_std)
% x: x coordinates
% y: either just one y vector, or 2xN or 3xN matrix of y-data
% fstr: format ('r' or 'b--' etc)
% std: number of standard deviations to display
% 

if nargin<4
    n_std = 2;
else
    n_std = n_std/sqrt(size(y,1));
end
ll=size(y,1);
% if size(y,1)>size(y,2)
%     y=y';
% end
 
if size(y,1)==1 % just plot one line
    plot(x,y,fstr);
end
 
if size(y,1)==2 %plot shaded area
    px=[x,fliplr(x)]; % make closed patch
    py=[y(1,:), fliplr(y(2,:))];
    patch(px,py,1,'FaceColor',fstr,'EdgeColor','none');
end
%  
% % [y_1,i1]=max(y(:,19));
% % [y_2,i2]=min(y(:,50));
%  [sz1,sz2]=size(y);
%  i1 = 1;
%  i2 = 5;
% y_mean = y(3,:);
y_mean = mean(y, 1);
y_std = std(y);

if size(y,1)>=3 % also draw mean
    px=[x,fliplr(x)];
    py=[y_mean+n_std*y_std, fliplr(y_mean-n_std*y_std)];
    patch(px,py,1,'FaceColor',fstr,'EdgeColor','none');
    hold on
  %  plot(x,y(ceil((1+ll)/2),:),fstr,'LineWidth',1.2);
    plot(x,y_mean,fstr,'LineWidth',1.4);
end
 hold off
alpha(.2); % make patch transparent