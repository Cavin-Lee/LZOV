function [accCount,para,timecount] = LZV(Xs,Ys,Xt,Yt,options)
% Leave Zero Out for Model Selection
% Input
% Xs Ys Xt Yt          s trainning data t testing data
% options.model     return trained model i.e. W for LR, nV for SVM
% options.predict    return predict label
% options.t              AugumentType  1 linear transform 2 mix up
% options.nT           number of Validation
% options.nP           number of Augument
% options.wtag        tag for weighting  0 unweight 1 weight
% options.Crange    nC*1 cell  nC is the number of hyper parameter
% Output
% accCount             accuracy in each iteration
% para                    selecting time of each parameter
% timecount            timecost

if nargin<5
    options.model=@LinearSVMModel; %model training
    options.predict=@LinearSVMPredict;% model predict
    options.nP=1;
    options.nT=100;
    options.wtag=0;
    options.vtag=0;
    options.t=1;
    options.Crange=[{2.^[-5:15]}];
end
if ~isfield(options,'Crange')
    error('Crange is needed')
end
C_num=1;
for i = 1:length (options.Crange)
    C_num=C_num*length(options.Crange{i});
end
para=zeros(C_num,1);
labSet = unique(Ys);
i=1;

for i=1: options.nT
    tic
    XV=[];YV=[];W=[];
    
    for j = 1:options.nP
        %%   Validataion Data Augumentation
        if options.t==1
            A = rand(size(Xs,2));
            A=orth(A);
            X_tmp = Xs*A;
            Y_tmp =Ys;
            XV = [XV ;X_tmp];
            YV = [YV ;Y_tmp];
        end
        if options.t==2  %mix up
            labSet = unique(Ys);
            X_tmp=[];
            Y_tmp=[];
            for iC=1:length(labSet)
                Ca_num=sum(Ys==labSet(iC));
                X_c=Xs(Ys==labSet(iC),:);
                D=dist(X_c,X_c');
                D(logical(eye(size(D))))=inf;
                [~,ind]= min(D);
                alpha=max(0.001,rand(Ca_num,1));
             X_tmp=[X_tmp; diag(alpha)*X_c+diag(1-alpha)*X_c(ind,:)];
            Y_tmp=[Y_tmp; labSet(iC)+ones(Ca_num,1)];               
            end
            XV = [XV ;X_tmp];
            YV = [YV ;Y_tmp];;
        end
        
        
        %% Weight Computing
        W_tmp = ones(length(Y_tmp),1);
        
        if options.wtag ==1
            
            
            for iC = 1:length(labSet)
                Va = X_tmp(Y_tmp==labSet(iC),:);
                Tr = Xs(Ys==labSet(iC),:);
                Y = [ones(size(Tr,1),1) ;zeros(size(Va,1),1)];
                X = [Tr ;Va];
                theta = glmfit(X, [Y ones(length(Y),1)], 'binomial', 'link', 'logit');
                P_tr=1./(1+exp(theta(1) - theta(2:end)'*Va'));
                P_va = 1-P_tr;
                W_tmp(Y_tmp==labSet(iC))= (size(Va,2)/size(Tr,2))*(P_tr./P_va);
            end
        end
        W = [W;W_tmp];
    end
    W(isinf(W))=1;
    %%   model estimation and validation
    
    for iC = 1:C_num
        Select_C = [];
        count_num = iC;
        for k = 1:length (options.Crange)-1
            tmp=1;
            for p=k+1:length (options.Crange)
                tmp=tmp*length(options.Crange{p});
            end
            Select_num = floor((count_num-1)/tmp)+1;
            count_num = mod((count_num-1),tmp)+1;
            Select_C = [Select_C options.Crange{k}(Select_num)];
        end
        Select_C = [Select_C options.Crange{length (options.Crange)}(count_num)];
        
        % model estimation
        model(iC) = options.model(Xs,Ys,Select_C);
        predict = options.predict(XV,model(iC));
        if options.vtag==0 % as one validation data
            acc(iC)=(predict==YV)'*W;
        end
        if options.vtag==1 % count
            Vali_Num=size(Xs,1);
            for v_iter =1:options.nP
                ind_v=(v_iter-1)*Vali_Num+1:v_iter*Vali_Num;
                acc(iC,v_iter)=(predict(ind_v)==YV(ind_v))'*W(ind_v);
            end
        end
        
    end
    %% model selection and testing
    if options.vtag==0
        ind = find(acc==max(acc(:)));
    end
    account_vali=zeros(C_num,1);
    if options.vtag==1
        for v_iter =1:options.nP
            ind_vali=find(acc(:,v_iter)==max(acc(:,v_iter)));
            account_vali(ind_vali)=account_vali(ind_vali)+1;
        end
        ind = find(account_vali==max(account_vali(:)));
    end
    para(ind(end))=para(ind(end))+1;
    
    predict = options.predict(Xt,model(ind(end)));
    accCount(i)=sum(predict==Yt)/length(Yt);
    %fprintf('%d iteration, acc = %d \n',i,accCount(i))
    
    timecount(i)=toc;
end