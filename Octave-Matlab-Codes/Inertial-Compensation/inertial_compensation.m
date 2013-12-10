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

p=size(fpdata_cal);
Nframes2=p(1,1);

%=====================================================================
% 1: CALIBRATION (Determining coefficients of correction matrix)
%=====================================================================

%---------------------------------------------------------------------
%Acceleration Matrix (D) Generation
%---------------------------------------------------------------------

   %Forming the Sparse Matrix
      a13=ones(Nframes2,1);
      acceldata_cal=[acceldata_cal a13];
      D=zeros(6*Nframes2,78);
         for i=1:Nframes2
            for j=1:6
               row=6*(i-1)+j;
               col=13*(j-1)+(1:13);
               D(row,col)=acceldata_cal(i,:);
            end
         end

%----------------------------------------------------------------------
%Force Matrix (B)Generation
%----------------------------------------------------------------------

    %Split data into FP1 and FP2
        fpdata_cal1=fpdata_cal(:,1:6);
        fpdata_cal2=fpdata_cal(:,7:12);
    %Reshaping for Least Squares
        B1=reshape(fpdata_cal1',6*Nframes2,1);
        B2=reshape(fpdata_cal2',6*Nframes2,1);

%----------------------------------------------------------------------
%Creating the Coefficients of the Correction Matrices (FP1/FP2)
%----------------------------------------------------------------------

C1=D\B1;
C2=D\B2;

%=========================================================================
%2: CORRECTION (Applying Calibration Matrices to Raw Force Data)
%=========================================================================

%----------------------------------------------------------------------
%Acceleration Matrix (D) Generation
%----------------------------------------------------------------------

      a13=ones(Nframes2,1);
      acceldata_cor=[acceldata_cor a13];
      D=zeros(6*Nframes2,78);
         for i=1:Nframes2
            for j=1:6
               row=6*(i-1)+j;
               col=13*(j-1)+(1:13);
               D(row,col)=acceldata_cor(i,:);
            end
         end

%----------------------------------------------------------------------
%Force Matrix (B)Generation
%----------------------------------------------------------------------

    %Split data into FP1 and FP2
        fpdata_cor1=fpdata_cor(:,1:6);
        fpdata_cor2=fpdata_cor(:,7:12);
    %Reshaping for Least Squares
        B1=reshape(fpdata_cor1',6*Nframes2,1);
        B2=reshape(fpdata_cor2',6*Nframes2,1);

%-----------------------------------------------------------------------
%Correcting the Force Data from Calibration Matrices (C1 and C2)
%-----------------------------------------------------------------------

    %Correcting the Forces and Moments
        B1c=B1-(D*C1);
        B1cr=reshape(B1c,6,Nframes2);
        B2c=B2-(D*C2);
        B2cr=reshape(B2c,6,Nframes2);

%=======================================================================
%3. COORDINATE TRANSFORMATION Rotating Force Vectors to Reference Frame
%=======================================================================

%Initial Reference Coordinate Position
   x=[marker_cor(1,1:3); marker_cor(1,4:6); marker_cor(1,7:9);...
      marker_cor(1,10:12); marker_cor(1,13:15)];
%Rearranging Force and Moment Vectors
    FMP1_corr=reshape(B1cr,6,Nframes2)';
    FMP2_corr=reshape(B2cr,6,Nframes2)';
    FP1_corr=FMP1_corr(:,1:3); MP1_corr=FMP1_corr(:,4:6);
    FP2_corr=FMP2_corr(:,1:3); MP2_corr=FMP2_corr(:,4:6);
    FP1_p=reshape(FP1_corr(:,1:3)',3*Nframes2,1);
    MP1_p=reshape(MP1_corr(:,1:3)',3*Nframes2,1);
    FP2_p=reshape(FP2_corr(:,1:3)',3*Nframes2,1);
    MP2_p=reshape(MP2_corr(:,1:3)',3*Nframes2,1);
%Determining the R and P Matrices
    R=[]; xpos=[]; RMS=[];
    for i=1:Nframes2
         y=[marker_cor(i,1:3); marker_cor(i,4:6); marker_cor(i,7:9);...
            marker_cor(i,10:12); marker_cor(i,13:15)];
         [R1,xpos1,RMS1]=soder(x,y);
         R=[R;R1];
         xpos=[xpos;xpos1];
         RMS=[RMS;RMS1];
    end
%Rotating the Force and Moment Vectors
    FP1=[];MP1=[];FP2=[];MP2=[];
    for i=1:3:Nframes2*3;
         FP1r=R(i:i+2,:)*FP1_p(i:i+2,:);
         MP1r=(cross(xpos(i:i+2,:),FP1r))+(R(i:i+2,:)*MP1_p(i:i+2,:));
         FP2r=R(i:i+2,:)*FP2_p(i:i+2,:);
         MP2r=(cross(xpos(i:i+2,:),FP2r))+(R(i:i+2,:)*MP2_p(i:i+2,:));
         FP1=[FP1;FP1r]; MP1=[MP1;MP1r];
         FP2=[FP2;FP2r]; MP2=[MP2;MP2r];
    end

%=========================================================================
%Generating Output
%=========================================================================

%Rearranging Matrices
    FP1=reshape(FP1,3,Nframes2)';
    MP1=reshape(MP1,3,Nframes2)';
    FP2=reshape(FP2,3,Nframes2)';
    MP2=reshape(MP2,3,Nframes2)';
%Compensated Forces
    comp_fp=[FP1 MP1 FP2 MP2];

end
