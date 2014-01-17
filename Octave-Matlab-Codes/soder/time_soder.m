a = rand(6, 3, 60000);
b = rand(6, 3, 60000);
tic()
rotations = zeros(3, 3, 60000);
translations = zeros(3, 60000);
for i = 1:size(a, 3)
    [R, d, ~] = soder(a(:, :, i), b(:, :, i));
    rotations(:, :, i) = R;
    translations(:, i) = d;
end
toc()
