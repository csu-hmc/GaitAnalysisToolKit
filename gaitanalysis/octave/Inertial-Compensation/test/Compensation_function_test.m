%Tests Inertial Compensation 

clc
clear

filename1='9.10.2013Stationary0001.txt';
[t1,f1,c1,d1]=CarenRead(filename1);
p=size(d1);
Nframes=p(1:1); 

%----------
%Filtering 
%----------

[num,den]=butter(2,6/(100/2));
df=filter(num,den,d1(:,91:136));
df=[d1(:,1:90) df];

%Removing First Second after Applying Filter
t1=t1(101:Nframes,:);
df=df(101:Nframes,:);
f1=f1(101:Nframes,:);
d1=d1(101:Nframes,:);

delay=0.072;
df2=df(:,1:120);
s1=df(:,121:136);
[t,f,d2]=time_delay(t1,f1,df2,s1,delay);

%Separating Matrices for Input
fpdata_cal=[d2(:,94:99) d2(:,103:108)];
acceldata_cal=[d2(:,122:124) d2(:,126:128) d2(:,130:132) d2(:,134:136)];
marker_data=d2(:,1:15);


[comp_fp]=inertial_compensation(fpdata_cal,acceldata_cal,...
                                marker_data,fpdata_cal,...
                                acceldata_cal);
                             
%----------------
%Plot Comparison
%----------------

%FP1 Forces
figure(1)
subplot(3,4,1)
plot(d1(:,94),'k')
hold on
plot(comp_fp(:,1),'r'); title('Force_F_P_1'); ylabel('X');
subplot(3,4,5)
plot(d1(:,95),'k')
hold on
plot(comp_fp(:,2),'r'); ylabel('Y');
subplot(3,4,9)
plot(d1(:,96),'k')
hold on
plot(comp_fp(:,3),'r'); ylabel('Z');

%FP1 Moments
subplot(3,4,2)
plot(d1(:,97),'k')
hold on
plot(comp_fp(:,4),'r'); title('Moment_F_P_1');
subplot(3,4,6)
plot(d1(:,98),'k')
hold on
plot(comp_fp(:,5),'r'); 
subplot(3,4,10)
plot(d1(:,99),'k')
hold on
plot(comp_fp(:,6),'r');

%FP2 Forces
subplot(3,4,3)
plot(d1(:,103),'k')
hold on
plot(comp_fp(:,7),'r'); title('Force_F_P_2'); 
subplot(3,4,7)
plot(d1(:,104),'k')
hold on
plot(comp_fp(:,8),'r'); 
subplot(3,4,11)
plot(d1(:,105),'k')
hold on
plot(comp_fp(:,9),'r'); xlabel('Samples'); 

%FP2 Moments
subplot(3,4,4)
plot(d1(:,106),'k')
hold on
plot(comp_fp(:,10),'r'); title('Moment_F_P_2'); 
subplot(3,4,8)
plot(d1(:,107),'k')
hold on
plot(comp_fp(:,11),'r'); 
subplot(3,4,12)
plot(d1(:,108),'k')
hold on
plot(comp_fp(:,12),'r'); 

