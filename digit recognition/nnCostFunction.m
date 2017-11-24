function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
          
%J = 0;
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));

a2 = sigmoid([ones(m, 1) X] * Theta1');
%5000x25
a3 = sigmoid([ones(m, 1) a2] * Theta2');
%5000x10
ym=zeros(size(y,1),num_labels);
%5000x10
for i=1:m
    ym(i,y(i))=1;
end

y=ym(:);
a=a3(:);

J=(-y'*log(a)-(1-y)'*log(1-a))/m + (sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);

d3=(a3-ym);
%5000x10

aa=(d3*Theta2(:,2:end));
bb=sigmoidGradient([ones(size(X,1),1) X]*Theta1');

d2=aa.*bb;
%5000x25
delta1=d2'*[ones(size(X,1),1) X];
delta2=d3'*[ones(size(a2,1),1) a2];

Theta1(:,1)=0;
Theta2(:,1)=0;
Theta1_grad=delta1/m+Theta1*lambda/m;
%25x401
Theta2_grad=delta2/m+Theta2*lambda/m;
%10x26

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
