function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cvec = [0.01 0.03 0.1 0.3 1 3 10 30];
Svec = [0.01 0.03 0.1 0.3 1 3 10 30];

Cbest = Cvec(1);
Sbest = Svec(1);
err_min = length(yval);
% Outer loop over all C values
for i = 1:length(Cvec)
    C = Cvec(i);
    % Inner loop over all sigma values
    for j = 1:length(Svec)
        sigma = Svec(j);
        % Train the SVM classifier using the training set (X,y)
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        
        % Use the cross-validation set to determine the best C and sigma
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        
        % If overall cross-validation error is lower, use these parameters
        if err < err_min
            err_min = err;
            Cbest = C;
            Sbest = sigma;
        end
    end
end

C = Cbest
sigma = Sbest



% =========================================================================

end
