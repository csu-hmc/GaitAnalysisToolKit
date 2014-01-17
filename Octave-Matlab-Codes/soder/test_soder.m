function assert = test_soder()

num_frames = 50; % n
num_markers = 5; % m

time = linspace(0, 10, num_frames);
frequency = 0.1 * 2 * pi;

% Create three Euler angles which vary with time.
% n x 3
euler_angles = [0.5 * sin(frequency .* time);
                1.0 * sin(frequency .* time);
                2.0 * sin(frequency .* time)]';

s = sin(euler_angles);
c = cos(euler_angles);

% These are the rotation matrices we are trying to identify, size: 3 x 3 x
% n. They represent the 123 Euler rotation of the second reference frame, B,
% with respect to the first reference frame, A, where va = R * vb.
expected_rotation = zeros(3, 3, num_frames);
for i = 1:num_frames
    % This is the direction cosine matrix for a 1-2-3 Euler rotation (body
    % fixed rotation). See Kane, Likins, Levinson (1983) Page 423.
    expected_rotation(:, :, i) = ...
        [ c(i, 2) .* c(i, 3),                                 -c(i, 2) .* s(i, 3),                                 s(i, 2);
          s(i, 1) .* s(i, 2) .* c(i, 3) + s(i, 3) .* c(i, 1), -s(i, 1) .* s(i, 2) .* s(i, 3) + c(i, 3) .* c(i, 1), -s(i, 1) .* c(i, 2);
         -c(i, 1) .* s(i, 2) .* c(i, 3) + s(i, 3) .* s(i, 1),  c(i, 1) .* s(i, 2) .* s(i, 3) + c(i, 3) .* s(i, 1),  c(i, 1) .* c(i, 2)];
end

% We assume that there are five markers located in the B body fixed
% reference frame. Thus, the markers' relative positions do not vary through
% time.
marker_initial_vectors_in_local_frame = rand(3, num_markers); % 3 x m

% The location of the marker set in the global reference frame varies with
% time in a simple linear fashion, size: 3 x n.
expected_translation = bsxfun(@times, [1.0; 2.0; 3.0], time);

% We now can express the vector to each marker in the global reference frame
% as the markers translate and rotate through time.
marker_trajectories = zeros(num_markers, 3, num_frames);
for i = 1:num_frames
    for j = 1:num_markers
        % 3 x 1 = 3 x 1 + 3 x 3 * 3 x 1
        marker_trajectories(j, :, i) = expected_translation(:, i) + ...
            expected_rotation(:, :, i) * marker_initial_vectors_in_local_frame(:, j);
    end
end

% Now use the soder function to compute the rotation matrices and
% translations vectors given the marker trajectories.
translation = zeros(3, num_frames);
rotation = zeros(3, 3, num_frames);
for i = 1:num_frames
    [R, q, ~] = soder(marker_trajectories(:, :, 1),
                      marker_trajectories(:, :, i));
    rotation(:, :, i) = R;
    translation(:, i) = q;
end

assert_rotation = all((rotation - expected_rotation) < 1e-12);
assert_translation = all((translation - expected_translation) < 1e-12);
assert = assert_translation && assert_rotation;

% TODO : Test the RMS.
