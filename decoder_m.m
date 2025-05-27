clear;
clc;

filename = 'encoded4.bin';  % zakładam m=3
fid = fopen(filename, 'rb');

r = 10;
N = 256;
m = 8;
mLvl = m^2;
nrBits = m; % liczba bitów na indeks

frame = 1;
a = ones(1, 10);
e = zeros(1, N);
while ~feof(fid) && frame <= 896
    % Odczytaj eMax
    eMax(frame, 1) = fread(fid, 1, 'float32');
    if numel(eMax(frame, 1)) ~= 1
        break;
    end
    
    a(frame, :) = fread(fid, 10, 'float32');
    if numel(a(frame, :)) ~= 10
        break;
    end

    e(frame, :) = fread(fid, 256, 'ubit2');  % Oczekujemy długości 256 zamiast 276
    if numel(e(frame, :)) ~= 256
        break;
    end
    
    eMin = -eMax(frame);
    delta = (eMax(frame) - eMin) / (mLvl - 1);

    indices(frame, :) = (e(frame, :) .* delta) + eMin;
    
    frame = frame + 1;
end

indices = indices(:, 10:256);
size(indices)
y = [];
prev_samples = zeros(1, 10);

for seg = 1:size(indices, 1)
    segment = zeros(1, 256);
    for j = 1:length(segment)
        if j <= 10
            if seg == 1
                input = [zeros(1, 11-j), segment(1:j-1)];
            else
                input = [prev_samples(11-j:10), segment(1:j-1)];
            end
        else
            input = segment(j-10:j-1);
        end
        if length(input) < 10
            input = [zeros(1, 10-length(input)), input];
        elseif length(input) > 10
            input = input(end-9:end);
        end
        prediction = -sum(a(seg, :) .* fliplr(input));
        segment(j) = prediction + indices(seg, min(j, size(indices, 2)));

    end
    prev_samples = segment(end-9:end);

    if isempty(y)
        y = segment;
    else
        overlap_start = length(y) - 9;
        overlap_end = length(y);
        y(overlap_start:overlap_end) = 0.5 * (y(overlap_start:overlap_end) + segment(1:10));
        y = [y, segment(11:end)];
    end
end

audiowrite('decoded.wav', y', 11025);
fclose(fid);
