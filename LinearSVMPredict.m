function predict = LinearSVMPredict(X,model)
% X data  n*d
% Y lable  n*1
Y = ones(size(X,1),1);
[predict_label, ~, ~] = svmpredict(Y,X, model, '-q');
predict=predict_label;