function [y, yd, ydd] = myfiltfilt(t, x, f0)

% performs low pass filtering and differentiation
% method is the same as used in HBM but bidirectional to eliminate lag

% Inputs
%	t (Nsamples x 1)       		Time stamps
%	x (Nsamples x Nchannels)	Input signals
%	f0 (scalar)					Corner frequency
%
% Outputs
%	y (Nsamples x Nchannels)		Filtered signal
%	yd (Nsamples x Nchannels)		First derivative
%	ydd (Nsamples x Nchannels)		Second derivative

	C = 0.802;						% correction factor for dual pass 2nd order filter (Winter book)
	y = filter_batch(-flipud(t), flipud(x), f0/C);		% filter backwards in time
	[y, yd, ydd] = filter_batch(t, flipud(y), f0/C);	% filter forward in time

end
%===================================================================================
function [y, yd, ydd] = filter_batch(t,x,f0)
	% filters a time series and also returns derivatives
	% uses real-time second order Butterworth filter (rtfilter.m)
	
	% some constants we will need
	a = (2*pi*f0)^2;
	b = sqrt(2)*(2*pi*f0);

	% allocate memory for the results
	n = size(x,1);
	y = zeros(size(x));
	yd = zeros(size(x));
	ydd = zeros(size(x));
	
	% Integrate the filter state equation using the midpoint Euler method with step h
	% initial conditions are y=0 and yd=0
	for i = 2:n
		h = t(i)-t(i-1);		% time step
		denom = 4 + 2*h*b + h^2*a;
		A = (4 + 2*h*b - h^2*a)/denom;
		B = 4*h/denom;
		C = -4*h*a/denom;
		D = (4 - 2*h*b - h^2*a)/denom;
		E = 2*h^2*a/denom;
		F = 4*h*a/denom;
		y(i)  = A*y(i-1) + B*yd(i-1) + E*(x(i)+x(i-1))/2;
		yd(i) = C*y(i-1) + D*yd(i-1) + F*(x(i)+x(i-1))/2;
		ydd(i) = (yd(i)-yd(i-1))/h;
	end
end
%===================================================================================
