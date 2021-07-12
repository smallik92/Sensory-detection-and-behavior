function [V,M,I]= simple_PCA(X)

X = bsxfun(@minus, X, mean(X,1));   
C = (X'*X)./(size(X,1)-1);  
[V, eval]=eig(C);
lambda=diag(eval);

[M,I]=sort(lambda,'descend');
V=V(:,I);