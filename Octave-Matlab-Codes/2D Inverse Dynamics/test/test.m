function [pass] = test
	% test the leg2d analysis code
	
	addpath('..');			% add parent folder to search path, this is where leg2d.m is located
	
	% load the raw data and extract time stamps, mocap data, and force plate data
	d = load('rawdata.txt');
	times = d(:,1);
	mocapdata = d(:,2:13);
	bodymass = 67.2;					% this data was from subject 7 from the HBM paper
	fpdata = d(:,14:16)/bodymass;		% normalize force plate data to body mass (as required by leg2d)
	
	% do the analysis
	options.freq = 6;		% cutoff frequency
	[angles, velocities, moments, forces] = leg2d(times, mocapdata, fpdata, options);
	
	% make the plots
	jointnames = {'hip','knee','ankle'};
	figure(1)
	clf
	for i=1:3
		subplot(5,3,i)
		plot(times,angles(:,i)*180/pi);
		ylabel('angle (deg)');
		xlabel('time (s)');
		title(jointnames{i});
	
		subplot(5,3,3+i)
		plot(times,moments(:,i));
		ylabel('moment (Nm/kg)');
		xlabel('time (s)');
	
		subplot(5,3,6+i)
		plot(times,moments(:,i).*velocities(:,i));
		ylabel('power (W/kg)');
		xlabel('time (s)');
	
		subplot(5,3,9+i)
		plot(times,velocities(:,i)*180/pi);
		ylabel('ang.vel (deg/s)');
		xlabel('time (s)');
	
		subplot(5,3,12+i)
		plot(times,forces(:,2*i-1:2*i));
		ylabel('force (N/kg)');
		xlabel('time (s)');
		legend('Fx','Fy');
	end
	
	% compare to correct reference results, and decide pass if max difference is below 1e-6
	allresults = [angles velocities moments forces];
	if (exist('refresults.mat') == 2)
		load('refresults.mat');
	else
		refresults = allresults;
		save('refresults.mat', 'refresults');
		fprintf('File refresults.mat has been created.');
	end
	pass = (max(max(abs(allresults - refresults))) < 1e-6);
	
end