function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m,1) X]; % 5000* 401

Z2 = X*Theta1'; %(5000*25) = (5000*401)*(401*25)
H2 = sigmoid(Z2); % (5000*25)

Z3 = [ones(size(H2,1),1) H2] * Theta2'; %(5000*10) = (5000*26)*(26*10)
H3 = sigmoid(Z3); %(5000*10)

[max_v, p] = max(H3,[],2);
% from every value of [1:10] max func finds max_value as max_v and its index as p


% look we are taking 5000 every time to understand but we are using "vectorised implementing" so,
%  we r basically going to next row ie for 5000 rows implicitly.
% so we ve to understand that we are performing for 1 row and rest its implemented by its own. 

% =========================================================================


end
