clear; close all; clc;

% Parameters
input_file = 'pan_tadeusz1.wav';
N = 256; % Segment length
r = 10; % Filter order
fs = 11025; % Sampling frequency
quant_bits = 2:8; % Quantization bits to process

% Read input audio
try
    [input, fs] = audioread(input_file);
catch
    error('Failed to read input file %s. Ensure it exists and is a valid WAV file.', input_file);
end
input = input(:,1); % Ensure mono signal
original_size = dir(input_file).bytes;

% Window function (same for encoder and decoder)
window = 0.5 * (1 - cos(2 * pi * (1:N) / (N + 1)));

% Loop over quantization bits
for m = quant_bits
    fprintf('Processing m=%d bits...\n', m);
    mLvl = 2^m; % Number of quantization levels
    
    % --- Encoder ---
    % Split input into overlapping segments
    step = N - r;
    num_segments = floor((length(input) - r) / step) + 1;
    splited = zeros(num_segments, N);
    for i = 1:num_segments
        start_idx = (i - 1) * step + 1;
        end_idx = min(start_idx + N - 1, length(input));
        segment = input(start_idx:end_idx)';
        if length(segment) < N
            segment(end+1:N) = 0; % Pad with zeros
        end
        splited(i, :) = segment;
    end
    
    % Apply windowing
    flattened = splited .* window;
    
    % Extend segments with zeros for correlation
    extended = zeros(num_segments, N + 2 * r);
    extended(:, r + 1:N + r) = flattened;
    
    % Levinson-Durbin for AR coefficients
    a = zeros(num_segments, r);
    sigma = zeros(num_segments, 1);
    for i = 1:num_segments
        R = xcorr(extended(i, :), r, 'biased');
        R = R(r + 1:end);
        if length(R) ~= r + 1
            warning('Autocorrelation vector R for segment %d has incorrect length: %d', i, length(R));
            continue;
        end
        try
            [a_temp, sigma_temp, ~] = L_D(R, r);
            if length(a_temp) == r && isscalar(sigma_temp)
                a(i, :) = a_temp;
                sigma(i) = sigma_temp;
            else
                warning('L_D returned incorrect sizes for segment %d: a=%d, sigma=%d', i, length(a_temp), length(sigma_temp));
                a(i, :) = zeros(1, r); % Fallback to zeros
                sigma(i) = 1e-10;
            end
        catch err
            warning('L_D failed for segment %d: %s', i, err.message);
            a(i, :) = zeros(1, r);
            sigma(i) = 1e-10;
        end
    end
    
    % Calculate residual errors and quantize
    eMax = zeros(num_segments, 1);
    e = zeros(size(splited));
    indices = zeros(size(splited));
    output_filename = ['encoded' int2str(m) '.bin'];
    output = fopen(output_filename, 'wb');
    
    for i = 1:num_segments
        % Compute residual errors
        for j = 1:N
            if j > r
                prev_samples = splited(i, j-1:-1:j-r);
                e(i, j) = splited(i, j) + sum(a(i, :) .* prev_samples);
            else
                e(i, j) = splited(i, j);
            end
        end
        eMax(i) = max(abs(e(i, :))) + 1e-10; % Avoid zero eMax
        
        % Quantization (mid-tread)
        eMin = -eMax(i);
        delta = (eMax(i) - eMin) / mLvl;
        indices(i, :) = round((e(i, :) - eMin) / delta - 0.5);
        indices(i, :) = max(0, min(indices(i, :), mLvl - 1));
        
        % Write to binary file
        fwrite(output, eMax(i), 'float32');
        fwrite(output, a(i, :), 'float32');
        fwrite(output, indices(i, :), ['ubit' num2str(m)]);
    end
    fclose(output);
    
    % Calculate compression ratio
    file_info = dir(output_filename);
    if file_info.bytes == 0
        warning('Encoded file %s is empty. Skipping decoding.', output_filename);
        continue;
    end
    encoded_size = file_info.bytes;
    compression_ratio = original_size / encoded_size;
    fprintf('m=%d bits: Compression ratio = %.2f:1\n', m, compression_ratio);
    
    % --- Decoder ---
    fid = fopen(output_filename, 'rb');
    if fid == -1
        warning('Failed to open %s for reading.', output_filename);
        continue;
    end
    window_inv = 1 ./ (window + 1e-10); % Inverse window
    
    frame = 1;
    eMax = [];
    a = [];
    e = [];
    max_frames = num_segments; % Expected number of segments
    while ~feof(fid) && frame <= max_frames
        % Read eMax
        temp = fread(fid, 1, 'float32');
        if isempty(temp)
            break;
        end
        eMax(frame, 1) = temp;
        
        % Read AR coefficients
        a_temp = fread(fid, r, 'float32');
        if length(a_temp) ~= r
            break;
        end
        a(frame, :) = a_temp;
        
        % Read quantized indices
        e_temp = fread(fid, N, ['ubit' num2str(m)]);
        if length(e_temp) ~= N
            break;
        end
        e(frame, :) = e_temp;
        
        % Reconstruct residual errors
        eMin = -eMax(frame);
        delta = (eMax(frame) - eMin) / mLvl;
        e(frame, :) = (e(frame, :) + 0.5) * delta + eMin;
        
        frame = frame + 1;
    end
    fclose(fid);
    
    if isempty(eMax)
        warning('No valid data read from %s. Skipping decoding.', output_filename);
        continue;
    end
    
    % Reconstruct signal
    y = [];
    prev_samples = zeros(1, r);
    for seg = 1:size(e, 1)
        segment = zeros(1, N);
        for j = 1:N
            if j <= r
                if seg == 1
                    input = [zeros(1, r - j + 1), segment(1:j-1)];
                else
                    input = [prev_samples(r - j + 1:end), segment(1:j-1)];
                end
            else
                input = segment(j-r:j-1);
            end
            if length(input) < r
                input = [zeros(1, r - length(input)), input];
            elseif length(input) > r
                input = input(end-r+1:end);
            end
            prediction = -sum(a(seg, :) .* fliplr(input));
            segment(j) = prediction + e(seg, j);
        end
        % Apply inverse windowing
        segment = segment .* window_inv;
        prev_samples = segment(end-r+1:end);
        
        % Overlap-add
        if isempty(y)
            y = segment;
        else
            overlap_start = length(y) - r + 1;
            overlap_end = length(y);
            y(overlap_start:overlap_end) = y(overlap_start:overlap_end) .* (1 - window(1:r)) + segment(1:r) .* window(1:r);
            y = [y, segment(r+1:end)];
        end
    end
    
    % Normalize output
    if ~isempty(y)
        y = y / (max(abs(y)) + 1e-10) * 0.95;
        audiowrite(['decoded_' int2str(m) '.wav'], y', fs);
        fprintf('Decoded audio saved as decoded_%d.wav\n', m);
    else
        warning('No audio reconstructed for m=%d.', m);
    end
end

% Levinson-Durbin function
function [a, sigma, k] = L_D(pi, r)
    a = zeros(r, 1);
    k = zeros(r, 1);
    sigma_vec = zeros(r + 1, 1);
    
    sigma_vec(1) = pi(1) + 1e-10; % Regularization
    if abs(sigma_vec(1)) < 1e-10
        sigma = 1e-10;
        return;
    end
    
    k(1) = -pi(2) / sigma_vec(1);
    a(1) = k(1);
    sigma_vec(2) = sigma_vec(1) * (1 - k(1)^2);
    
    for i = 2:r
        sum_k = 0;
        for j = 1:i-1
            sum_k = sum_k + a(j) * pi(i - j + 1);
        end
        k(i) = -(pi(i + 1) + sum_k) / sigma_vec(i);
        a_old = a(1:i-1);
        for j = 1:i-1
            a(j) = a_old(j) + k(i) * conj(a_old(i - j));
        end
        a(i) = k(i);
        sigma_vec(i + 1) = sigma_vec(i) * (1 - k(i)^2);
    end
    sigma = sigma_vec(r + 1); % Return final sigma value
end