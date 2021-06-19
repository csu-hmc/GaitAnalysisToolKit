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
    d = load([script_dir filesep 'input.tsv']);

    times = d(:,1);

    mocapdata = d(:, 2:13);
    % the GTRO marker has lots of dropout in this data, set them to nan
    nan_tol = 1e-10;
    mocapdata((-nan_tol < mocapdata) & (mocapdata < nan_tol)) = nan;

    bodymass = 77.0;
    fpdata = d(:, 14:16) / bodymass; % normalize force plate data to body mass

    % do the analysis
    options.freq = 6.0; % cutoff frequency
    [angles, velocities, moments, forces] = ...
        leg2d(times, mocapdata, fpdata, options);

    % scale the loads back to normal
    moments = bodymass .* moments;
    forcess = bodymass .* forces;

    expected_ang = load('output_angles.tsv');
    expected_ang = expected_ang(:, 2:4);

    expected_vel = load('output_rates.tsv');
    expected_vel = expected_vel(:, 2:4);

    expected_mom = load('output_torques.tsv');
    expected_mom = expected_mom(:, 2:4);

    % TODO : See if Ton has these forces.
    %expected_for = load('output_forces.tsv');
    %expected_for = expected_for(:, 2:4);

    if do_plot

        jointnames = {'hip','knee','ankle'};
        figure(1)

        for i=1:3

            subplot(5, 3, i)
            plot(times, rad2deg(expected_ang(:, i)), 'k', ...
                 times, rad2deg(angles(:, i)), 'b.');
            ylabel('angle (deg)');
            title(jointnames{i});

            subplot(5, 3, 3 + i)
            plot(times, rad2deg(expected_vel(:, i)), 'k', ...
                 times, rad2deg(velocities(:, i)), 'b.');
            ylabel('ang.vel (deg/s)');

            subplot(5, 3, 6 + i)
            plot(times, expected_mom(:,i), 'k', ...
                 times, moments(:,i), 'b.');
            ylabel('moment (Nm)');

            % TODO : Octave doesn't plot this row for some reason.
            subplot(5, 3, 9+i)
            plot(times, expected_mom(:,i) .* expected_vel(:,i), 'k', ...
                 times, moments(:,i) .* velocities(:,i), 'b.');
            ylabel('power (W)');

            % TODO : plot the joint force comparisons
            %subplot(5, 3, 12+i)
            %plot(times, expected_for(:, 2*i-1:2*i), 'k', ...
                 %times, forces(:, 2*i-1:2*i), 'b');
            %ylabel('force (N/kg)');
            xlabel('time (s)');
        end
    end

    % compare to correct reference results, and decide pass if max
    % difference is below 1e-6
    results = [angles velocities moments];
    expected_results = [expected_ang expected_vel expected_mom];
    tol = 1e-6;
    if exist('OCTAVE_VERSION', 'builtin') ~= 0
        % Octave has an extended version of assert with floating point
        % comparisons.
        assert(results, expected_results, tol);
    else
        assert(all(all(abs(results - expected_results) < tol)));
    end

end

function deg = rad2deg(rad)
    deg = rad * 180.0 / pi;
end
