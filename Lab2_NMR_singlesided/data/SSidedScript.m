%%
CPMG1 = readtable('CPMG (1)\data.csv');
CPMG2 = readtable('CPMG (2)\data.csv');
CPMG3 = readtable('CPMG (3)\data.csv');

fnc = @(b,x) exp(b(1) + b(2).*x);

name="CPMG 1"
mdlT1_trans = fitlm(CPMG1{:,"Var1"},log(CPMG1{:,"Var2"})) %CPMG(1), time is in microseconds
T1_trans=(1/(-1*mdlT1_trans.Coefficients.Estimate(2)))*10^-03; % T2 in MILLISECONDS (1)
T1_trans(1,2) = abs((mdlT1_trans.Coefficients.SE(2)/mdlT1_trans.Coefficients.Estimate(2))*(1/(-1*mdlT1_trans.Coefficients.Estimate(2)))*10^-03);
M01_trans=exp(mdlT1_trans.Coefficients.Estimate(1));


figure
hold on
name="CPMG 2"
mdlT2_trans = fitlm(CPMG2{:,"Var1"},log(CPMG2{:,"Var2"}./max(CPMG2{:,"Var2"}))) %CPMG(2)
T2_trans=(1/(-1*mdlT2_trans.Coefficients.Estimate(2)))*10^-03;
T2_trans(1,2) = abs((mdlT2_trans.Coefficients.SE(2)/mdlT2_trans.Coefficients.Estimate(2))*(1/(-1*mdlT2_trans.Coefficients.Estimate(2)))*10^-03);
M02_trans=exp(mdlT2_trans.Coefficients.Estimate(1));
params_ = mdlT2_trans.Coefficients.Estimate;
params_upper = mdlT2_trans.Coefficients.Estimate + 1.96.*mdlT2_trans.Coefficients.SE;
params_lower = mdlT2_trans.Coefficients.Estimate - 1.96.*mdlT2_trans.Coefficients.SE;
scatter(CPMG2{:,"Var1"}*10^-03,CPMG2{:,"Var2"}/max(CPMG2{:,"Var2"}),Marker="x",MarkerEdgeColor="k")
plot(CPMG2{:,"Var1"}*10^-03,fnc(params_,CPMG2{:,"Var1"}),'b-',LineWidth=0.3)
plot(CPMG2{:,"Var1"}*10^-03,fnc(params_upper,CPMG2{:,"Var1"}),'r--',LineWidth=0.3)
plot(CPMG2{:,"Var1"}*10^-03,fnc(params_lower,CPMG2{:,"Var1"}),'r--',LineWidth=0.3)
xlabel("$T_{E}$ $ (ms)$",Interpreter="latex")
ylabel("Normalized Signal Intensity",Interpreter="latex")
title("Middle Layer CPMG Curve",Interpreter="latex" )
grid()


hold on
name="CPMG 3"
mdlT3_trans = fitlm(CPMG3{:,"Var1"},log(CPMG3{:,"Var2"})) %CPMG(3)
T3_trans=(1/(-1*mdlT3_trans.Coefficients.Estimate(2)))*10^-03;
T3_trans(1,2) = abs((mdlT3_trans.Coefficients.SE(2)/mdlT3_trans.Coefficients.Estimate(2))*(1/(-1*mdlT3_trans.Coefficients.Estimate(2)))*10^-03);
M03_trans=exp(mdlT3_trans.Coefficients.Estimate(1));
%% 
SSE1 = readtable('SSE (1)\data.dat');
SSE2 = readtable('SSE (2)\data.dat');
SSE3 = readtable('SSE (3)\data.dat');

g = 14; % mouse gradient, T/m
gamma = 42.576*10^6; % gyromag ratio of 1H, Hz/T
delta = 1; % ms, diffusion time

name="SSE 1"
x_axis = (gamma*2*pi*g)^2.*((2/3)*(SSE1{:,"Var1"}*10^-3).^3 + delta*10^-3.*(SSE1{:,"Var1"}*10^-3).^2); % x variable in m^-2*s, norm signal is linearly proportional to it
signal = log(SSE1{:,"Var2"}/SSE1{1,"Var2"}); %normalized signal
mdlD1 = fitlm(x_axis,signal,'y~x1-1') 
D1=mdlD1.Coefficients.Estimate(1); %  in meters^2/s (1)
D1(1,2) = mdlD1.Coefficients.SE(1);

figure
hold on
name="SSE 2"
x_axis = (gamma*2*pi*g)^2.*((2/3)*(SSE2{:,"Var1"}*10^-3).^3 + delta*10^-3.*(SSE2{:,"Var1"}*10^-3).^2); % x variable, norm signal is linearly proportional to it
signal = log(SSE2{:,"Var2"}/SSE2{1,"Var2"}); %normalized signal
mdlD2 = fitlm(x_axis,signal,'y~x1-1') 
D2=mdlD2.Coefficients.Estimate(1); %  in MILLISECONDS (1)
D2(1,2) = mdlD2.Coefficients.SE(1);
plot(mdlD2,Marker="x",MarkerEdgeColor="k")
xlabel("$\gamma^2 g^2 \delta^2(\frac{2}{3} \delta + \Delta)$ $(\frac{s}{m^2})$",Interpreter="latex")
ylabel("Log(Normalized Signal Intensity)",Interpreter="latex")
title("Middle Layer SSE Curve",Interpreter="latex" )
grid()

hold off

name="SSE 3"
x_axis = (gamma*2*pi*g)^2.*((2/3)*(SSE3{:,"Var1"}*10^-3).^3 + delta*10^-3.*(SSE3{:,"Var1"}*10^-3).^2); % x variable, norm signal is linearly proportional to it
signal = log(SSE3{:,"Var2"}/SSE3{1,"Var2"}); %normalized signal
mdlD3 = fitlm(x_axis,signal,'y~x1-1') 
D3=mdlD3.Coefficients.Estimate(1); %  in MILLISECONDS (1)
D3(1,2) = mdlD3.Coefficients.SE(1);

%% 
SR1 = readtable('SRmouse (1)\data.dat');
SR2 = readtable('SRmouse (2)\data.dat');
SR3 = readtable('SRmouse (3)\data.dat');

fnc = @(b,x) b(1).*(1-exp(b(2).*x)) + b(3);

name="SR 1"
B0 = [0.6 -1/400 0];
mdlT1_long = fitnlm(SR1{:,"Var1"}*10^3,SR1{:,"Var2"},fnc,B0)
T1_long=abs(1/mdlT1_long.Coefficients.Estimate(2)); %  in MILLISECONDS (1)
T1_long(1,2) = abs((mdlT1_long.Coefficients.SE(2)/mdlT1_long.Coefficients.Estimate(2))*(1/abs(mdlT1_long.Coefficients.Estimate(2))));


figure
hold on
name = "SR 2"
B0 = [0.6 -1/400 0];
mdlT2_long = fitnlm(SR2{:,"Var1"}*10^3,SR2{:,"Var2"}/max(SR2{:,"Var2"}),fnc,B0)
T2_long=abs(1/mdlT2_long.Coefficients.Estimate(2)); %  in MILLISECONDS (1)
T2_long(1,2) = abs((mdlT2_long.Coefficients.SE(2)/mdlT2_long.Coefficients.Estimate(2))*(1/abs(mdlT2_long.Coefficients.Estimate(2))));
params_ = mdlT2_long.Coefficients.Estimate;
params_upper = mdlT2_long.Coefficients.Estimate + 1.96.*mdlT2_long.Coefficients.SE;
params_lower = mdlT2_long.Coefficients.Estimate - 1.96.*mdlT2_long.Coefficients.SE;
scatter(SR2{:,"Var1"}*10^3,SR2{:,"Var2"}/max(SR2{:,"Var2"}),Marker="x",MarkerEdgeColor="k")
plot(SR2{:,"Var1"}*10^3,fnc(params_,SR2{:,"Var1"}*10^3),'b-',LineWidth=0.3)
plot(SR2{:,"Var1"}*10^3,fnc(params_upper,SR2{:,"Var1"}*10^3),'r--',LineWidth=0.3)
plot(SR2{:,"Var1"}*10^3,fnc(params_lower,SR2{:,"Var1"}*10^3),'r--',LineWidth=0.3)
xlabel("$t_{rec}$ $(ms)$",Interpreter="latex")
ylabel("Normalized Signal Intensity",Interpreter="latex")
title("Middle Layer SR Curve",Interpreter="latex" )
grid()

hold off

figure
hold on
name="SR 3"
B0 = [1 -1/300 0];
mdlT3_long = fitnlm(SR3{:,"Var1"}*10^3,SR3{:,"Var2"}./max(SR3{:,"Var2"}),fnc,B0)
T3_long=abs(1/mdlT3_long.Coefficients.Estimate(2)); %  in MILLISECONDS (1)
T3_long(1,2) = abs((mdlT3_long.Coefficients.SE(2)/mdlT3_long.Coefficients.Estimate(2))*(1/abs(mdlT3_long.Coefficients.Estimate(2))));
params_ = mdlT3_long.Coefficients.Estimate;
params_upper = mdlT3_long.Coefficients.Estimate + 1.96.*mdlT3_long.Coefficients.SE;
params_lower = mdlT3_long.Coefficients.Estimate - 1.96.*mdlT3_long.Coefficients.SE;
scatter(SR3{:,"Var1"}*10^3,SR3{:,"Var2"}./max(SR3{:,"Var2"}),Marker="x",MarkerEdgeColor="k")
plot(SR3{:,"Var1"}*10^3,fnc(params_,SR3{:,"Var1"}*10^3),'b-',LineWidth=0.3)
plot(SR3{:,"Var1"}*10^3,fnc(params_upper,SR3{:,"Var1"}*10^3),'r--',LineWidth=0.3)
plot(SR3{:,"Var1"}*10^3,fnc(params_lower,SR3{:,"Var1"}*10^3),'r--',LineWidth=0.3)
xlabel("$t_{rec}$ $ (ms)$",Interpreter="latex")
ylabel("Normalized Signal Intensity",Interpreter="latex")
title("Superficial Layer SR Curve",Interpreter="latex" )
grid()

hold off
%%
PR1 = readtable('Profile_B22FA\1\Profile_B22FA.dat');
plot(PR1{:,"Depth"},PR1{:,"Sig"}./max(PR1{:,"Sig"}),'r-x',LineWidth=0.3,MarkerSize=5,MarkerEdgeColor='k')
xlabel("Depth $(\mu m)$",Interpreter="latex")
ylabel("Normalized Signal Intensity",Interpreter="latex")
title("NMR Signal Profile",Interpreter="latex" )
xlim([50 3150])
grid()



