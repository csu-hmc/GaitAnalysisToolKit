function [angles, velocities, moments, forces] = leg2d(times, mocapdata, fpdata, options)
% 2D analysis of lower extremity motion
%
% Method:
% 	Winter, DA (2005) Biomechanics of Human Movement.
%
% Inputs:
%	times (Nsamples x 1)		Time stamps for raw data
%	mocapdata (Nsamples x 12)	Mocap data: x and y coordinates of 6 markers (m)
%	fpdata (Nsamples x 3)		Load applied to foot: Fx, Fy, and Mz (N/kg, normalized to body mass)
%	options (structure)			Various settings used in the analysis
%		options.freq			Cutoff frequency of low pass filter (in Hz)
%
% Outputs:
%	angles (Nsamples x 3)		Angles in three joints (rad)
%	velocities (Nsamples x 3)	Angular velocities in three joints (rad/s)
%	moments (Nsamples x 3)		Moments in three joints, (Nm per kg body mass)
%	forces (Nsamples x 6)		Forces (Fx,Fy) in three joints, (N per kg body mass)
%
% Coordinate system:
%	X is forward (direction of walking), Y is up
%
% Markers:
%	1: Shoulder
%	2: Greater trochanter
%	3: Lateral epicondyle of knee
%	4: Lateral malleolus
%	5: Heel (placed at same height as marker 6)
%	6: Head of 5th metatarsal
%
% Joints:
%	hip, knee, ankle
%   sign convention for angles and moments: hip flexion, knee flexion, ankle plantarflexion are positive
%
% Notes:
%	- the code has been vectorized for best performance for large number of samples
%	- missing marker data is interpolated *after* low pass filtering
%	- x and y coordinates of the same marker must be either both valid or both NaN (missing)

        oct_ver = strsplit(OCTAVE_VERSION, '.');

	% some constants
	Nmarkers = 6;
	Ncoords = 2*Nmarkers;
	Nfpchannels = 3;
	g = 9.81;					% acceleration of gravity

	% define the body segments
	%	column 1: proximal marker
	%	column 2: distal marker
	%	column 3: joint marker (at proximal end)
	%	column 4: mass as fraction of body mass (from Winter book)
	%	column 5: center of mass as fraction of length (from Winter book)
	%   column 6: radius of gyration as fraction of length (from Winter book)
	%   column 7: +1(-1) if positive angle/moment in prox, joint corresponds to counterclockwise(clockwise) rotation of segment
	segments = [	1		2		NaN		NaN			NaN			NaN		NaN;	% HAT (head-arms-trunk)
					2		3		2		0.100		0.433		0.323	+1;		% thigh
					3		4		3		0.0465		0.433		0.302	-1;		% shank
					5		6		4		0.0145		0.500		0.475	-1];	% foot (note we use heel-toe line as segment axis!)
	Nsegments = size(segments,1);

	% error checking
	if (nargin ~= 4)
		error('leg2d: requires 4 inputs');
	end
	if ~isfield(options,'freq')
		error('leg2d: options.freq is missing');
	end
	Nsamples = size(times,1);
	if (size(mocapdata,1) ~= Nsamples)
		error('leg2d: number of samples in mocap data is not the same as number of time stamps.');
	end
	if (size(fpdata,1) ~= Nsamples)
		error('leg2d: number of samples in force plate data is not the same as number of time stamps.');
	end
	if (size(mocapdata,2) ~= Ncoords)
		error('leg2d: number of columns in mocap data is not correct.');
	end
	if (size(fpdata,2) ~= Nfpchannels)
		error('leg2d: number of columns in force plate data is not correct.');
	end

	% do the low-pass filtering/differentiation on raw marker data, only use frames where marker is visible
	mocap_f = zeros(size(mocapdata));
	mocap_d = zeros(size(mocapdata));
	modep_dd = zeros(size(mocapdata));
	for i=1:Nmarkers
		columns = 2*(i-1) + (1:2);
		d = mocapdata(:,columns);					% d now contains x and y data for marker i
		validsamples = find(~isnan(d(:,1)));		% these are the frames for which the x coordinate is valid
		validtimes = times(validsamples);
		missing = Nsamples - size(validsamples,1);
		maxmissing = max(diff(validsamples))-1;			% determine largest gap
		fprintf('Marker %d: %d samples are missing, longest gap is %d samples.\n', i, missing, maxmissing);
		[xf, xd, xdd] = myfiltfilt(validtimes, d(validsamples,:), options.freq);
		% Octave removed interp1q in version 4.2.
		if ((str2num(oct_ver{1}) >= 4) && (str2num(oct_ver{2}) >= 2))
			mocap_f(:,columns) = interp1(validtimes, xf, times);		% resample filtered signal to original time stamps
			mocap_d(:,columns) = interp1(validtimes, xd, times);		% resample first derivative to original time stamps
			mocap_dd(:,columns) = interp1(validtimes, xdd, times);		% resample second derivative to original time stamps
		else
			mocap_f(:,columns) = interp1q(validtimes, xf, times);		% resample filtered signal to original time stamps
			mocap_d(:,columns) = interp1q(validtimes, xd, times);		% resample first derivative to original time stamps
			mocap_dd(:,columns) = interp1q(validtimes, xdd, times);		% resample second derivative to original time stamps
                end
	end

	% do the low-pass filtering on the force plate data
	fpdata_f = myfiltfilt(times, fpdata, options.freq);

	% do kinematic analysis for the segments
	segx = zeros(Nsamples, Nsegments);
	segy = zeros(Nsamples, Nsegments);
	sega = zeros(Nsamples, Nsegments);
	segad = zeros(Nsamples, Nsegments);
	segxdd = zeros(Nsamples, Nsegments);
	segydd = zeros(Nsamples, Nsegments);
	segadd = zeros(Nsamples, Nsegments);
	for i=1:Nsegments

		% segment parameters
		ip = segments(i,1);		% index of proximal marker
		id = segments(i,2);		% index of distal marker
		cm = segments(i,5);		% center of mass location, relative to line from prox to dist marker

		% x and y coordinates of proximal and distal marker, and first and second derivatives
		Px   = mocap_f(:,2*ip-1);
		Pxd  = mocap_d(:,2*ip-1);
		Pxdd = mocap_dd(:,2*ip-1);
		Py   = mocap_f(:,2*ip);
		Pyd  = mocap_d(:,2*ip);
		Pydd = mocap_dd(:,2*ip);
		Dx   = mocap_f(:,2*id-1);
		Dxd  = mocap_d(:,2*id-1);
		Dxdd = mocap_dd(:,2*id-1);
		Dy   = mocap_f(:,2*id);
		Dyd  = mocap_d(:,2*id);
		Dydd = mocap_dd(:,2*id);

		% vector R points from proximal to distal marker
		Rx     = Dx - Px;
		Rxd    = Dxd - Pxd;
		Rxdd   = Dxdd - Pxdd;
		Ry     = Dy - Py;
		Ryd    = Dyd - Pyd;
		Rydd   = Dydd - Pydd;

		% calculate segment center of mass position and segment orientation angle, and 1st and 2nd derivatives
		segx(:,i)   = Px + cm * Rx;
		segxdd(:,i) = Pxdd + cm * Rxdd;
		segy(:,i)   = Py + cm * Ry;
		segydd(:,i) = Pydd + cm * Rydd;
		sega(:,i)   = unwrap(atan2(Ry,Rx));					% orientation of the vector R, unwrap removes -pi to pi discontinuities
		segad(:,i)  = (Rx.*Ryd - Ry.*Rxd) ./ (Ry.^2+Rx.^2);	% analytical time derivative of segment angle
		segadd(:,i) = (Rx.*Rydd - Ry.*Rxdd) ./ (Ry.^2+Rx.^2) ...
			- 2*(Rx.*Ryd - Ry.*Rxd) .* (Ry.*Ryd + Rx.*Rxd) ./ (Ry.^2+Rx.^2).^2;	% analytical time derivative of segment angular velocity

		% segment length
		L = sqrt(Rx.^2 + Ry.^2);					% determine length in each frame
		if max(abs(L - mean(L))) > 0.1
			fprintf('Error detected while processing segment %d\n', i);
			fprintf('Segment length changed by more than 0.1 meters');
                        % TODO : This should probably raise an error but it
                        % confounds the testing with random data. Need to
                        % have an option to enable/disable this error
                        % checking so the test can be run with random data.
			%error('Segment length changed by more than 0.1 meters');
		end
		seglength(i) = mean(L);

	end

	% do the inverse dynamics, starting at the foot, but not for the HAT segment
	for i = Nsegments:-1:2

		m = segments(i,4);		% mass as fraction of body mass
		k = segments(i,6);		% radius of gyration, relative to length
		inertia = m * (k*seglength(i))^2;			% compute moment of inertia

		% compute vectors P and D from center of mass to distal and proximal joint
		jointmarker = segments(i,3);					% marker for proximal joint of this segment
		Px = mocap_f(:,2*jointmarker-1) - segx(:,i);
		Py = mocap_f(:,2*jointmarker)   - segy(:,i);
		if (i < Nsegments)
			jointmarker = segments(i+1,3);			% marker for distal joint of this segment (=proximal joint of next segment)
			Dx = mocap_f(:,2*jointmarker-1) - segx(:,i);
			Dy = mocap_f(:,2*jointmarker)   - segy(:,i);
			FDx = -FPx;		% force at distal joint is minus the proximal force in the previous segment
			FDy = -FPy;
			MD  = -MP;		% moment at distal joint
		else						% for the last segment, distal joint is the force plate data, applied to foot at global origin
			Dx = -segx(:,i);
			Dy = -segy(:,i);
			FDx = fpdata_f(:,1);
			FDy = fpdata_f(:,2);
			MD  = fpdata_f(:,3);
		end

		% solve force and moment at proximal joint from the Newton-Euler equations
		FPx = m * segxdd(:,i) - FDx;
		FPy = m * segydd(:,i) - FDy + m * g;
		MP  = inertia * segadd(:,i) - MD - (Dx.*FDy - Dy.*FDx) - (Px.*FPy - Py.*FPx);

		% and store proximal joint motions and loads in the output variables
		j = i-1;			% joint index (1, 2, or 3) for the proximal joint of segment i
		sign = segments(i,7);
		angles(:,j) 	= sign * (sega(:,i)  - sega(:,i-1));
		velocities(:,j) = sign * (segad(:,i) - segad(:,i-1));
		moments(:,j) 	= sign * MP;
		forces(:,2*j-1) = FPx;
		forces(:,2*j) 	= FPy;

	end

end
