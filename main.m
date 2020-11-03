% clear
% clc
warning off
addpath(genpath('FUN'))
datapath = '/home/cavin/Experiment/LZO/UCI';
file = dir(fullfile(datapath,'*.txt'));
tmp=cell(4,length(file));
for f = 1:length(file)
    pause
    namefile(f)={(file(f).name)};
    
    data = load([datapath '/' file(f).name]);
    X = data(:,1:end-1);Y=data(:,end);
    clear data
    Cs(f,1) = length(unique(Y));
    Cs(f,2) =size(X,2);
    Cs(f,3) = length(Y);
    
    fprintf('-------\n')

    ratio = 0.5;
    lab = unique(Y);
    Xs = [];Xt=[];Ys=[];Yt=[];
    for i = 1:length(lab)
        C_num = sum(Y == lab(i));
        C_trnum = fix(ratio*C_num);
        X_c = X(Y == lab(i),:);
        Xs = [Xs ;X_c(1:C_trnum,:)];
        Ys = [Ys; i*ones(C_trnum,1)];
        Xt = [Xt; X_c(C_trnum+1:end,:)];
        Yt = [Yt; i*ones(C_num-C_trnum,1)];
    end
    disp([file(f).name ' is start'])
    
    %% Kfold
%     options.model=@RBFSVMModel; %model training
%     options.predict=@RBFSVMPredict;% model predict
%     options.k=10;
%     options.nT=100;
%     options.Crange=[{2.^[-5:15]} {2.^[-5:15]} ];
%     [accCount,para,timecount] = KfoldCV(Xs,Ys,Xt,Yt,options);
%     tmp(1,f)=num2cell(mean(accCount));
%     tmp(2,f)=num2cell(std(accCount));
%     tmp(3,f)=num2cell(mean(timecount));
%     tmp(4,f)=num2cell(std(timecount));
%     
%     csvwrite('Kfold_100_RBF_result.xls',tmp)
%     save('Kfold_RBF.mat','accCount' ,'para','timecount')
    % LZV
% % %     
        options.model=@LinearSVMModel;
        options.predict=@LinearSVMPredict;
        options.nP=1;     % augmentation mumber
        options.nT=100;  % repetition number
        options.wtag=0;  % 0 unweight 1 weight
        options.t=2;        % 1 linear transform 2 mix-up
        options.Crange=[{2.^[-5:15]}];
        options.vtag=1;
        [accCount,para,timecount] = LZV(Xs,Ys,Xt,Yt,options);
        tmp(1,f)=num2cell(mean(accCount));
        tmp(2,f)=num2cell(std(accCount));
        tmp(3,f)=num2cell(mean(timecount));
        tmp(4,f)=num2cell(std(timecount));
        tmp(5,f)=namefile(f);
        csvwrite('LZV_2T1P0Wnew_result.xls',tmp)
       save('LZV_2T1P0W_new.mat','accCount' ,'para','timecount')

end