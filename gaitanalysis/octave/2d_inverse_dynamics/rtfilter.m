function [y, yd, ydd] = rtfilter(t, x, f0)

% Real-time low pass Butterworth filter, processes one data sample at a time.
%
% Inputs:
%	t			time stamp of input x (s)
%	x			input data at time t (may be a scalar, array, or matrix)
%	f0			cutoff frequency (Hz)
%
% Outputs:
%	y			filtered x
%	yd			filtered derivative of x
%	ydd			filtered second derivative of x

	% Internal state of the filter must be preserved between function calls
	persistent state

	% If this is the first call to this function, or we have gone back in time, reset the filter
	if isempty(state) || (t <= state.t)
		y = x;
		yd = zeros(size(x));
		ydd = zeros(size(x));
	% Otherwise, solve the state equation by Euler integration
	else
		% Calculate time between current and previous sample
		h = t - state.t;
		
		% Compute coefficients of the state equation
		a = (2*pi*f0)^2;
		b = sqrt(2)*(2*pi*f0);
		
		% Integrate the filter state equation using the midpoint Euler method with step h
		denom = 4 + 2*h*b + h^2*a;
		A = (4 + 2*h*b - h^2*a)/denom;
		B = 4*h/denom;
		C = -4*h*a/denom;
		D = (4 - 2*h*b - h^2*a)/denom;
		E = 2*h^2*a/denom;
		F = 4*h*a/denom;
		y = A*state.y + B*state.yd + E*(x+state.x)/2;
		yd = C*state.y + D*state.yd + F*(x+state.x)/2;
		ydd = (yd-state.yd)/h;
		
	end
	
	% Store the filter state
	state.t = t;
	state.x = x;
	state.y = y;
	state.yd = yd;

end
