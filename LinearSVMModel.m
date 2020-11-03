function model = LinearSVMModel(X,Y,C)
% X data
% Y lable
% C hyperparameter
cmd = ['-t 0 -c ' num2str(C) ' -q'];% linear kernel
model= svmtrain(Y,X, cmd); % Linear Kernel