% soder.m: Matlab function to determine rigid body rotation & translation
% From:
% I. Soederqvist and P.A. Wedin (1993) Determining the movement of the skeleton
%   using well-configured markers. J. Biomech. 26:1473-1477.
% Same algorithm is described in:
% J.H. Challis (1995) A prodecure for determining rigid body transformation
%   parameters, J. Biomech. 28, 733-737.
% The latter also includes possibilities for scaling, reflection, and
% weighting of marker data.
%
% Written by Ron Jacobs (R.S. Dow Neurological Institute, Porland OR),
% adapted by Ton van den Bogert (University of Calgary).
%
% Input:
% x: 3-D marker coordinates in position 1 (3 columns, one row for each marker)
% y: 3-D marker coordinates in position 2 (same format)
%
% Output:
% R: rotation matrix
% d: translation vector
% rms: the root mean square fit error of the rigid body model
%
% the rigid body model is: y = R*x + d
%
function[R,q,rms]=soder(x,y)

[nmarkers,ndimensions]=size(x);
% we could give an error message if ndimensions is not 3

mx=mean(x);
my=mean(y);

% construct matrices A and B, subtract the mean so there is only rotation
for i=1:nmarkers,
  A(i,:)=x(i,:)-mx;
  B(i,:)=y(i,:)-my;
end
A = A';
B = B';

% The singular value decomposition to calculate R with det(R)=1
C=B*A';
[P,T,Q]=svd(C);
R=P*diag([1 1 det(P*Q')])*Q';

% Calculate the translation vector from the centroid of all markers
q=my'-R*mx';

% calculate RMS value of residuals
sumsq = 0;
for i=1:nmarkers
  ypred = R*x(i,:)' + q;
  sumsq = sumsq + norm(ypred-y(i,:)')^2;
end
rms = sqrt(sumsq/3/nmarkers);
