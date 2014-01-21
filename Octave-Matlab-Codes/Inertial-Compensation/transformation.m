function [rotated_forces]=transformation(markers,forces)
                                      
%=========================================================================
%FUNCTION transformation:
%   Rotates force vectors into a global reference frame using the
%   soderquist method
%
%--------
%Inputs:
%--------
%      markers       (Nsamples x 15)   XYZ positions of 5 reference plane
%                                      markers
%      forces        (Nsamples x 12)   3D force plate data (forces/moments)
%                                      for both force plates in the form: 
%                                      [FP1XYZ MP1XYZ FP2XYZ MP2XYZ]
%--------
%Outputs:
%--------
%      rotated_forces(Nsamples x 12)    Rotated 3D force plate data 
%                                      (forces/moments) for both force 
%                                       plates in the form:
%                                      [FP1XYZ MP1XYZ FP2XYZ MP2XYZ]
%=========================================================================

Nframes_cor=length(forces);
forces_FP1=forces(:,1:6); 
forces_FP2=forces(:,7:12); 

%-------------------------------------------------------------------------
% Make the soder.m and mmat.m files available to this function
%-------------------------------------------------------------------------
    path_to_this_file = mfilename('fullpath');
    [directory_of_this_file, ~, ~] = fileparts(path_to_this_file);
    addpath([directory_of_this_file filesep '..' filesep 'soder'])
    addpath([directory_of_this_file filesep '..' filesep 'mmat'])
%-------------------------------------------------------------------------
%Rearranging Data
%-------------------------------------------------------------------------
    %Initial Reference Coordinate Position
        x=[markers(1,1:3); markers(1,4:6); markers(1,7:9);...
           markers(1,10:12); markers(1,13:15)];
    %Rearranging Force and Moment Vectors
        FP1_corr=forces_FP1(:,1:3); MP1_corr=forces_FP1(:,4:6);
        FP2_corr=forces_FP2(:,1:3); MP2_corr=forces_FP2(:,4:6);
        FP1_p=reshape(FP1_corr(:,1:3)',3,1,Nframes_cor);
        MP1_p=reshape(MP1_corr(:,1:3)',3,1,Nframes_cor);
        FP2_p=reshape(FP2_corr(:,1:3)',3,1,Nframes_cor);
        MP2_p=reshape(MP2_corr(:,1:3)',3,1,Nframes_cor);
%-------------------------------------------------------------------------
%Determining the R and P Matrices
%-------------------------------------------------------------------------
        R=zeros(3,3,Nframes_cor); xpos=zeros(3,1,Nframes_cor);
        for i=1:Nframes_cor
             y=[markers(i,1:3); markers(i,4:6); markers(i,7:9);...
                markers(i,10:12); markers(i,13:15)];
            [R1,xpos1]=soder(x,y);
             R(:,:,i)=R1;
            xpos(:,:,i)=xpos1;
        end
    %Rotating the Force and Moment Vectors
        FP1=mmat(R,FP1_p);
        MP1=cross(xpos,FP1,1)+mmat(R,MP1_p);
        FP2=mmat(R,FP2_p);
        MP2=cross(xpos,FP2,1)+mmat(R,MP2_p);
%-------------------------------------------------------------------------
%Generating Output
%-------------------------------------------------------------------------
    %Rearranging Matrices
        FP1=reshape(FP1,3,Nframes_cor)';
        MP1=reshape(MP1,3,Nframes_cor)';
        FP2=reshape(FP2,3,Nframes_cor)';
        MP2=reshape(MP2,3,Nframes_cor)';
    %Compensated Forces
        rotated_forces=[FP1 MP1 FP2 MP2];
end 