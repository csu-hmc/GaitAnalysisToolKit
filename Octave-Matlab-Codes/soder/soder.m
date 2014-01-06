function [R, d, rms] = soder(x, y)
% function [R, d, rms] = soder(x, y)
%
% Returns the rotation matrix and translation vector between two point
% clouds of markers in 3D space that are assumed to be attached to the same
% rigid body in two different positions and orientations.
%
% Input
% -----
% x: double, size(n, 3)
%   The 3-D cartesion marker coordinates in the first position where the
%   columns are the x, y, and z coordinates for each marker.
% y: double, size(n, 3)
%   The 3-D cartesion marker coordinates in the second position where the
%   columns are the x, y, and z coordinates for each marker.
%
% Output
% ------
% R: double, size(3, 3)
%   This rotation matrix is defined such that va = R * vb where va is the
%   vector, v, expressed in the first reference frame and vb is the same
%   vector expressed in the second reference frame.
% d: double, size(3, 1)
%   The translation vector in the first reference frame.
% rms: double
%   The root mean square fit error of the rigid body model.
%
% Notes
% -----
%
% The rigid body model is: y = R * x + d
%
% This alogrithm is explicitly taken from:
%
% I. Soederqvist and P.A. Wedin (1993) Determining the movement of the
% skeleton using well-configured markers. J. Biomech. 26:1473-1477.
%
% But the same algorithm is described in:
%
% J.H. Challis (1995) A prodecure for determining rigid body transformation
% parameters, J. Biomech. 28, 733-737.
%
% The latter also includes possibilities for scaling, reflection, and
% weighting of marker data.
%
% This function was Written by Ron Jacobs (R.S. Dow Neurological Institute,
% Porland OR) and adapted by Ton van den Bogert (University of Calgary). It
% was further updated by Jason K. Moore (Unversity of Cleveland) in 2014.

[nmarkers, ndimensions] = size(x);

if size(x, 1) ~= size(y, 1)
    error('x and y must have the same number of markers.')
end

if size(x, 2) ~= 3 || size(y, 2) ~= 3
    error('x and y must have three coordinates for each marker.')
end

% x: n x 3
% y: n x 3

% This is the mean position of all markers in each reference frame.
mx = mean(x); % mx: 1 x 3
my = mean(y); % mx: 1 x 3

% Subtract the mean so that markers are now centered at the origin of each
% reference frame. Note: uses array broadcasting.
A = x - mx; % A: n x 3
B = y - my; % B: n x 3

% Use singular value decomposition to calculate the rotation matrix, R, with
% det(R) = 1.

[P, T, Q] = svd(B' * A); % 3 x 3

R = P * diag([1,  1, det(P * Q')]) * Q'; % 3 x 3

% Calculate the translation vector from the centroid of all markers. d is
% expressed in the same frame as y, i.e. the global reference frame.
d = my' - R * mx'; % 3 x 1

% calculate RMS value of residuals
sumsq = 0;
for i=1:nmarkers
  ypred = R * x(i, :)' + d;
  sumsq = sumsq + norm(ypred - y(i, :)')^2;
end
rms = sqrt(sumsq/ 3 / nmarkers);
