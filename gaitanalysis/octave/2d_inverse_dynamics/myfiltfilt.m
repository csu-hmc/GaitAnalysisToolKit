function [y, yd, ydd] = myfiltfilt(t, x, f0);

% performs low pass filtering and differentiation
% method is the same as used in HBM but bidirectional to eliminate lag

% Inputs
%	t (Nsamples x 1)       		Time stamps
%	x (Nsamples x Nchannels)	Input signals
%	f0 (scalar)					Corner frequency
%
% Outputs
%	y (Nsamples x 1)			Filtered signal
%	yd (Nsamples x 1)			First derivative
%	ydd (Nsamples x 1)			Second derivative

	C = 0.802;												% correction factor for dual pass 2nd order filter (Winter book)
	y = rtfilter_batch(-flipud(t), flipud(x), f0/C);		% filter backwards in time
	[y, yd, ydd] = rtfilter_batch(t, flipud(y), f0/C);		% filter forward in time

end
%===================================================================================
function [y, yd, ydd] = rtfilter_batch(t,x,f0)
	% filters a time series and also returns derivatives
	% uses real-time second order Butterworth filter (rtfilter.m)

	n = size(x,1);
	y = zeros(size(x));
	yd = zeros(size(x));
	ydd = zeros(size(x));
	
	for i = 1:n
		[y(i,:), yd(i,:), ydd(i,:)] = rtfilter(t(i),x(i,:),f0);
	end
	
end
%===================================================================================