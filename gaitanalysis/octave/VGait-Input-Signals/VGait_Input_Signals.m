%Sandra Hnat
%Cleveland State University
%6/7/2013

%==========================================================================
%This program generates white noise for the purpose of adding perturbation
%to test subjects while walking on the V-gait platform.  
%==========================================================================

clc
clear

%-------------------------------------------------------------------------
%Creating the Longitudinal Perturbation Signal 
%-------------------------------------------------------------------------
    %Declaring Variables
        fc1=[8.9 2.8 1.7 2 2.1];          %Cutoff Frequencies
        speed1=[0.8 1.2 1.6 2.5 3.25];    %Nominal Speed
        var1=[40 25 25 25 25];            %Variance of acceleration
    %Longitudinal Perturbations
        rand_speed_all=[]; ramp_speed_all=[]; ramp_speed_end_all=[];
        inputs=length(fc1);
        for i=1:5
            fc=fc1(:,i);
            speed=speed1(:,i);
            var=var1(:,i);
            sim('longitudinal_perturbation.mdl')
            sim('ramp_speed')
            random_speed_all(:,i)=random_speed;
            ramp_speed_all(:,i)=ramp_speed;
            ramp_speed_end_all(:,i)=flipdim(ramp_speed,1);
        end
%-------------------------------------------------------------------------
%Creating Times
%-------------------------------------------------------------------------
    %Create 30 Seconds of Preparation Time (Zeroing and Calibration Pose)
        t_start=0:0.0033:30;t_start=t_start';
        t_start_length=length(t_start);
        t_start_endtime=t_start(t_start_length,1);
        speed_start=zeros(t_start_length,1);
    %Starting Ramp Time
        t_ramp=time_ramp+t_start_endtime;
        t_ramp_length=length(t_ramp); 
        t_ramp_endtime=t_ramp(t_ramp_length);
    %Treadmill Belt Time
        t_belt=time_belt+t_ramp_endtime;
        t_belt_length=length(t_belt); 
        t_belt_endtime=t_belt(t_belt_length);
    %Ending Ramp Time
        t_ramp_end=t_ramp-t_start(t_start_length)+t_belt_endtime;
        t_ramp_end_length=length(t_ramp_end); 
        t_ramp_end_endtime=t_ramp_end(t_ramp_end_length);
        t_end=t_start+t_ramp_end(t_ramp_end_length);
        
Longitudinal_Perturbation=[t_start repmat(speed_start,1,inputs);...
                           t_ramp ramp_speed_all;...
                           t_belt random_speed_all; 
                           t_ramp_end ramp_speed_end_all;...
                           t_end repmat(speed_start,1,inputs)];
%-------------------------------------------------------------------------
%Creating the Lateral Perturbation Signal
%-------------------------------------------------------------------------
    sim('lateral_perturbation.mdl')
    Lateral_Perturbation=[time_vgait random_sway];
%-------------------------------------------------------------------------
%Plotting the Signals 
%-------------------------------------------------------------------------
    figure(1)
    c={'0.8 m/s','1.2 m/s','1.6 m/s','2.5 m/s','3.25 m/s'};
    maximum=[]; minimum=[]; avg=[]; std_all=[];
    for j=1:inputs
        subplot(5,1,j)
        plot(Longitudinal_Perturbation(:,1),Longitudinal_Perturbation(:,j+1))
        xlabel('Time (s)')
        ylabel('Velocity (m/s)')
        title(c{j},'Fontweight','bold')
        maximum1=max(random_speed_all(:,j));
        minimum1=min(random_speed_all(:,j));
        mean1=mean(random_speed_all(:,j));
        std1=std(random_speed_all(:,j));
        maximum=[maximum maximum1];
        minimum=[minimum minimum1];
        avg=[avg mean1];
        std_all=[std_all std1];
    end
    figure(2)
        plot(time_vgait,random_sway)
        xlabel('Time (s)')
        ylabel('Lateral Position (m)')
        title('Lateral Perturbation')
%-------------------------------------------------------------------------
%Saving Files
%-------------------------------------------------------------------------
%     filename_belt='Longitudinal_Perturbation.txt';
%     dlmwrite(filename_belt,Longitudinal_Perturbation)
%     filename_vgait='Lateral_Perturbation.txt';
%     dlmwrite(filename_vgait,Lateral_Perturbation)