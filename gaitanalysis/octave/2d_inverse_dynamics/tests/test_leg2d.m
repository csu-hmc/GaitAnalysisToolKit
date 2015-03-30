function test_leg2d(varargin)
    % Unit test for the leg2d analysis code.
    %
    % Parameters
    % ==========
    % do_plot : boolean, optional
    %     If 'true' a comparison plot will be displayed if 'false' only the
    %     test will run.

    if isempty(varargin)
        do_plot = false;
    else
        do_plot = varargin{1};
    end

    script_path = mfilename('fullpath');
    script_dir = fileparts(script_path);

    % add parent folder to the path, this is where leg2d.m is located
    addpath([script_dir filesep '..']);

    % load the raw data and extract time stamps, mocap data, and force plate
    % data
    d = load([script_dir filesep 'rawdata.txt']);
    times = d(:,1);
    mocapdata = d(:,2:13);
    bodymass = 67.2; % this data was from subject 7 from the HBM paper
    fpdata = d(:,14:16)/bodymass; % normalize force plate data to body mass

    % do the analysis
    options.freq = 6; % cutoff frequency
    [angles, velocities, moments, forces] = ...
        leg2d(times, mocapdata, fpdata, options);

    load([script_dir filesep 'refresults.mat']);

    expected_ang = refresults(:, 1:3);
    expected_vel = refresults(:, 4:6);
    expected_mom = refresults(:, 7:9);
    expected_for= refresults(:, 10:end);

    if do_plot
        jointnames = {'hip','knee','ankle'};
        figure(1)
        for i=1:3
            subplot(5, 3, i)
            plot(times, expected_ang(:,i)*180/pi, 'k', ...
                 times, angles(:,i)*180/pi, 'b.');
            ylabel('angle (deg)');
            xlabel('time (s)');
            title(jointnames{i});

            subplot(5, 3, 3+i)
            plot(times, expected_vel(:,i)*180/pi, 'k', ...
                 times, velocities(:,i)*180/pi, 'b.');
            ylabel('ang.vel (deg/s)');
            xlabel('time (s)');

            subplot(5, 3, 6+i)
            plot(times, expected_mom(:,i), 'k', ...
                 times, moments(:,i), 'b.');
            ylabel('moment (Nm/kg)');
            xlabel('time (s)');

            % TODO : Octave doesn't plot this row for some reason.
            subplot(5, 3, 9+i)
            plot(times, expected_mom(:,i) .* expected_vel(:,i), 'k', ...
                 times, moments(:,i) .* velocities(:,i), 'b.');
            ylabel('power (W/kg)');
            xlabel('time (s)');

            subplot(5, 3, 12+i)
            plot(times, expected_for(:, 2*i-1:2*i), 'k', ...
                 times, forces(:, 2*i-1:2*i), 'b');
            ylabel('force (N/kg)');
            xlabel('time (s)');
        end
    end

    % compare to correct reference results, and decide pass if max
    % difference is below 1e-6
    allresults = [angles velocities moments forces];
    tol = 1e-6;
    if exist('OCTAVE_VERSION', 'builtin') ~= 0
        % Octave has an extended version of assert with floating point
        % comparisons.
        assert(allresults, refresults, tol);
    else
        assert(all(all(abs(allresults - refresults) < tol)));
    end

end
