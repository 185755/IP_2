clear;
clc;

filename = 'encoded3.bin';  % zakładam m=3
fid = fopen(filename, 'rb');

r = 10;
N = 256;
m = 3;
mLvl = m^2;
nrBits = m; % liczba bitów na indeks


frame = 1;
a = ones(1, 10);
e = zeros(1, N + 2*r);
while ~feof(fid) & frame <= 896
    % Odczytaj eMax
    eMax(frame, 1) = fread(fid, 1, 'float32');
    if isempty(eMax)
        break;
    end
    
    a(frame, :) = fread(fid, 10, 'float32');
    if isempty(a)
        break;
    end

    e(frame, :) = fread(fid, 276, 'ubit2');

    if isempty(e)
        break;
    end
    
    eMin = -eMax(frame);
    delta = (eMax(frame) - eMin) / (mLvl - 1);

    indices(frame, :) = (e(frame, :) .* delta) + eMin;
    
    
    
    % Wyświetl podsumowanie ramki
    % fprintf('Frame %d:\n', frame);
    % fprintf('  eMax: %.10f\n', eMax(frame));
    % fprintf('  a: [%s]\n', num2str(a(frame)));
    % fprintf('  e: [%s]\n', num2str(e(frame, 1:10)));
    % plot(indices)
    
    frame = frame + 1;
end
indices = indices(:, 10 : 266)
y = [];
prev_samples = zeros(1, 10)

for seg = 1:length(e)
    segment = zeros(1, 256);
    for j = 1 : length(segment)
        if j <= 10
            if seg == 1
                 input = [zeros(1,11-j), segment(1:j-1)];
            else
                input = [prev_samples(11-j:10), segment(1:j-1)];
            end
        else
            input = segment(j-10:j-1);
        end
        if length(input) < 10
            input = [zeros(1,10-length(input)), input];
        elseif length(input) > 10
            input = input(end-9:end);
        end
        prediction = -sum(a(seg, :) .* fliplr(input));
        segment(j) = prediction + indices(seg, j);
    end
    prev_samples = segment(end-9:end);

    if isempty(y)
                y = segment;
    else
        % Overlap-add with previous segment
        overlap_start = length(y) - 9;
        overlap_end = length(y);
        y(overlap_start:overlap_end) = 0.5 * (y(overlap_start:overlap_end) + segment(1:10));
        y = [y, segment(11:end)];
    end
end
audiowrite('decoded.wav', y', 11025);
fclose(fid);
