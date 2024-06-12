T = readtable('CPMGalbume1B.dat');
sumDistrT2 = sum(T{:,"Sig_Np"},'omitnan');
weightsT2 = T{:,"Sig_Np"}./sumDistrT2;
T2 = sum(T{:,"T"}.*weightsT2(:),'omitnan')
figure
hold on

plot(T,"T","Sig_Np")
set(gca, 'XScale', 'log');
hold off
figure
hold on

for i = 1:numel(T{:,"Sig"})
    if T{i,"Sig"} <= 0
       T{i,"Sig"} = abs(T{i,"Sig"});
    end
end
xspace = linspace(eps(0),max(T{:,"SigT"}),1000);
mdl = fitlm(T{:,"SigT"},log(T{:,"Sig"}))
plot(xspace,exp(mdl.Coefficients.Estimate(1)+mdl.Coefficients.Estimate(2)*xspace),"LineStyle","-",'linewidth',4,'Color','r')
scatter(T,"SigT","Sig")
set(gca,'yscale','log')
hold off

T = readtable('CPMGtuorlo1B.dat');
sumDistrT2 = sum(T{:,"Sig_Np"},'omitnan');
weightsT2 = T{:,"Sig_Np"}./sumDistrT2;
T2 = sum(T{:,"T"}.*weightsT2(:),'omitnan')
figure
hold on

plot(T,"T","Sig_Np")
set(gca, 'XScale', 'log');
hold off
figure
hold on

for i = 1:numel(T{:,"Sig"})
    if T{i,"Sig"} <= 0
       T{i,"Sig"} = abs(T{i,"Sig"});
    end
end
xspace = linspace(eps(0),max(T{:,"SigT"}),1000);
mdl = fitlm(T{:,"SigT"},log(T{:,"Sig"}))
plot(xspace,exp(mdl.Coefficients.Estimate(1)+mdl.Coefficients.Estimate(2)*xspace),"LineStyle","-",'linewidth',4,'Color','r')
scatter(T,"SigT","Sig")
set(gca,'yscale','log')
hold off

T = readtable('IRalbumeH.dat');
sumDistrT1 = sum(T{:,"Sig_Np"},'omitnan');
weightsT1 = T{:,"Sig_Np"}./sumDistrT1;
T1 = sum(T{:,"T"}.*weightsT1(:),'omitnan')
figure
hold on

plot(T,"T","Sig_Np")
set(gca, 'XScale', 'log');
hold off
figure
hold on

fnc = @(b,x) b(1).*(1-(1+b(3))*exp(b(2).*x));
B0 = [1.5e06 -1/1000 0.8];
time = T{:,"SigT"};
signal = T{:,"Sig"};
time = time(1:64);
signal = -1*signal(1:64);
signal = signal - min(signal)/2;
scatter(time,signal)
mdl = fitnlm(time,signal,fnc,B0)

hold off

T = readtable('IRtuorloH.dat');
sumDistrT1 = sum(T{:,"Sig_Np"},'omitnan');
weightsT1 = T{:,"Sig_Np"}./sumDistrT1;
T1 = sum(T{:,"T"}.*weightsT1(:),'omitnan')
figure
hold on

plot(T,"T","Sig_Np")
set(gca, 'XScale', 'log');
hold off
figure
hold on

fnc = @(b,x) b(1).*(1-(1+b(3))*exp(b(2).*x));
B0 = [1.5e06 -1/1000 0.8];
time = T{:,"SigT"};
signal = T{:,"Sig"};
time = time(1:64);
signal = -1*signal(1:64);
signal = signal - min(signal)/2;
scatter(time,signal)
mdl = fitnlm(time,signal,fnc,B0)

hold off

