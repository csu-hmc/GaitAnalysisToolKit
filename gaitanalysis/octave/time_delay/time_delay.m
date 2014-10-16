function [t_adj,f_adj,data_adj]=time_delay(t,f,data_raw,accel,delay)

%=========================================================================
%FUNCTION time_delay:
%   1) Accounts for the delay between accelerometer and force plate signals
%   by adding a 72 ms delay to the accelerometer signals and interpolating 
%   the data.  
%   2) Removes all NaN from the interpolated data.  
%
%--------
%Inputs:
%--------
%    t         (Nsamples x 1)        Time vector
%    f         (Nsamples x 1)        Frame vector
%    data_raw  (Nsamples x 136)      Entire data matrix minus
%                                    accelerometer signals
%    accel     (Nsamples x 18)       Accelerometer signals (including EMG)
%    delay      scalar               The time difference between force and 
%                                    accelerometer signals
%--------
%Outputs:
%--------
%    t_adj     (Nsamples-adj x 1)    Adjusted time vector
%    f_adj     (Nsamples-adj x 1)    Adjusted frame vector
%    data_adj  (Nsamples-adj x 136)  Adjusted data matrix, with all signals
%                                    having the NaN portion of the 
%                                    interpolation removed, and the delay
%                                    subtracted from accelerometer signals 
%=========================================================================

%Adding Signal Delay 
      accel_adj=interp1(t,accel,t+delay);
%Removing all NaN from the Original Matrices
      Nframes=find(isnan(accel_adj), 1 )-1;
      data_adj=[data_raw(1:Nframes,:) accel_adj(1:Nframes,:)]; 
      t_adj=t(1:Nframes,1);
      f_adj=f(1:Nframes,1);
end