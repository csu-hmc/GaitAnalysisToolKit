function [comp_fp]=inertial_compensation(fpdata_cal,acceldata_cal,...
                                         marker_cor,fpdata_cor,...
                                         acceldata_cor) 
                                      
%=========================================================================
%FUNCTION inertial_compensation:
%   1)Compensates for the inertia of a moving instrumented treadmill, 
%     assuming a linear relationship between force plate signals and 
%     accelerometers
%   2)Generates a calibration matrix of coefficients based on linear least
%     squares regression between forces (B) and accelerations (D) of an
%     unweighted treadmill platform under prescribed movements
%   3)Corrects force signals of a weighted platform under similar movements
%     by applying the calibration matrix and subtracting inertial artifacts
%   4)Rotates the corrected force vectors into the global reference frame
%
%--------
%Inputs:
%--------
%  ~~Unweighted Treadmill (Calibration)~~
%      fpdata_cal    (Nsamples x 12)   3D force plate data (forces/moments)
%                                      for both force plates in the form:
%                                      [FP1XYZ MP1XYZ FP2XYZ MP2XYZ]
%      acceldata_cal (Nsamples x 12)   3D accelerations from 4 
%                                      accelerometers in the form:
%                                      [A1XYZ A2XYZ A3XYZ A4XYZ]
%  ~~Weighted Treadmill (Correction)~~
%      marker_cor    (Nsamples x 15)   XYZ positions of 5 reference plane
%                                      markers
%      fpdata_cor    (Nsamples x 12)   3D force plate data (forces/moments)
%                                      for both force plates in the form: 
%                                      [FP1XYZ MP1XYZ FP2XYZ MP2XYZ]
%      acceldata_cor (Nsamples x 12)   3D accelerations from 4 
%                                      accelerometers in the form:
%                                      [A1XYZ A2XYZ A3XYZ A4XYZ]
%--------
%Outputs:
%--------
%      comp_fp       (Nsamples x 12)   Compensated 3D force plate data 
%                                      (forces/moments) for both force 
%                                       plates in the form:
%                                      [FP1XYZ MP1XYZ FP2XYZ MP2XYZ]
%=========================================================================

%Compensation
    compensated_forces=compensation(fpdata_cal,acceldata_cal,...
                                           fpdata_cor,acceldata_cor);  
%Transformation
    rotated_forces=transformation(marker_cor,compensated_forces);
    comp_fp=rotated_forces;
end 