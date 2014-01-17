function [compensated_forces]=compensation(fpdata_cal,acceldata_cal,...
                                           fpdata_cor,acceldata_cor) 
                                      
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
%      fpdata_cor    (Nsamples x 12)   3D force plate data (forces/moments)
%                                      for both force plates in the form: 
%                                      [FP1XYZ MP1XYZ FP2XYZ MP2XYZ]
%      acceldata_cor (Nsamples x 12)   3D accelerations from 4 
%                                      accelerometers in the form:
%                                      [A1XYZ A2XYZ A3XYZ A4XYZ]
%--------
%Outputs:
%--------
%      compensated_forces (Nsamples x 12)   Compensated 3D force plate data 
%                                           (forces/moments) for both force 
%                                           plates in the form:
%                                           [FP1XYZ MP1XYZ FP2XYZ MP2XYZ]
%=========================================================================

Nframes_cal=length(fpdata_cal);
Nframes_cor=length(fpdata_cor);

%=====================================================================
% 1: CALIBRATION (Determining coefficients of correction matrix)
%=====================================================================

%---------------------------------------------------------------------
%Force Matrix (B) and Acceleration Matrix (D) Generation
%---------------------------------------------------------------------
      %Acceleration (D) Matrix
      a13=ones(Nframes_cal,1);
      acceldata_cal=[acceldata_cal a13];
      D=zeros(6*Nframes_cal,78);
         for i=1:Nframes_cal
            for j=1:6
               row=6*(i-1)+j;
               col=13*(j-1)+(1:13);
               D(row,col)=acceldata_cal(i,:);
            end
         end
    %Force Matrix (B)Generation
        fpdata_cal1=fpdata_cal(:,1:6);
        fpdata_cal2=fpdata_cal(:,7:12);
    %Reshaping for Least Squares
        B1=reshape(fpdata_cal1',6*Nframes_cal,1);
        B2=reshape(fpdata_cal2',6*Nframes_cal,1);
%----------------------------------------------------------------------
%Creating the Coefficients of the Correction Matrices (FP1/FP2)
%----------------------------------------------------------------------
    C1=D\B1;
    C2=D\B2;
%=========================================================================
%2: CORRECTION (Applying Calibration Matrices to Raw Force Data)
%=========================================================================

%----------------------------------------------------------------------
%Force Matrix (B) and Acceleration Matrix (D) Generation
%----------------------------------------------------------------------
      %Acceleration (D) Matrix
      a13=ones(Nframes_cor,1);
      acceldata_cor=[acceldata_cor a13];
      D=zeros(6*Nframes_cor,78);
         for i=1:Nframes_cor
            for j=1:6
               row=6*(i-1)+j;
               col=13*(j-1)+(1:13);
               D(row,col)=acceldata_cor(i,:);
            end
         end
      %Force Matrix (B)Generation
        fpdata_cor1=fpdata_cor(:,1:6);
        fpdata_cor2=fpdata_cor(:,7:12);
    %Reshaping for Least Squares
        B1=reshape(fpdata_cor1',6*Nframes_cor,1);
        B2=reshape(fpdata_cor2',6*Nframes_cor,1);
%-----------------------------------------------------------------------
%Correcting the Force Data from Calibration Matrices (C1 and C2)
%-----------------------------------------------------------------------
        B1c=B1-(D*C1);
        B1cr=reshape(B1c,6,Nframes_cor);
        B2c=B2-(D*C2);
        B2cr=reshape(B2c,6,Nframes_cor);
%-------------------------------------------------------------------------
%Generating Output
%-------------------------------------------------------------------------
        FP1=reshape(B1cr,6,Nframes_cor)';
        FP2=reshape(B2cr,6,Nframes_cor)';
        compensated_forces=[FP1 FP2]; 
end 