function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%returns the cost and gradient for the collaborative filtering problem.

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);
           
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

J=sum(sum(((X*Theta').*R-Y).^2))/2 + sum(sum(X.^2))*lambda/2 + sum(sum(Theta.^2))*lambda/2;

%Y=num_movies*num_users
for i=1:num_movies
    ind=find(R(i,:)==1);
    %ind=list of users that rated i movie
    Theta_t=Theta(ind,:);
    %Theta_t=attributes of ind users 
    Y_t=Y(i,ind);
    X_grad(i,:)=(X(i,:)*Theta_t'-Y_t)*Theta_t + lambda*X(i,:);
end

for j=1:num_users
    ind=find(R(:,j)==1);
    %ind=list of movies rated by user j
    X_t=X(ind,:);
    Y_t=Y(ind,j);
    Theta_grad(j,:)=(X_t*Theta(j,:)'-Y_t)'*X_t + lambda*Theta(j,:);
end
grad = [X_grad(:); Theta_grad(:)];
end
