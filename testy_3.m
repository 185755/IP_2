%% Main script to run encoder and decoder
clear; close all; clc;

% Run encoder
disp('Running encoder...');
predictive_encoder();

% Run decoder
disp('Running decoder...');
if ~exist('decoded', 'dir')
    mkdir('decoded');
end
predictive_decoder();

disp('Processing complete!');

% Encoder
function predictive_encoder()
    % Parameters
    N = 256;        % Segment length
    r = 10;         % AR model order and overlap between segments
    m_bits = [2, 3, 4]; % Quantization bits to test
    
    % Read WAV file
    [y, Fs] = audioread('pan_tadeusz1.wav');
    if Fs ~= 11025
        y = resample(y, 11025, Fs);
        Fs = 11025;
    end
    if size(y,2) > 1
        y = mean(y,2); % Convert to mono if stereo
    end
    y = y(:)'; % Ensure row vector
    
    % Added: Normalize input signal to [-1, 1] to maximize quantization range
    y = y / max(abs(y));
    
    % Added: Apply pre-emphasis filter to boost high frequencies (reduces audible noise)
    pre_emphasis_coeff = 0.95;
    y = filter([1, -pre_emphasis_coeff], 1, y);
    
    % Create output directory if it doesn't exist
    if ~exist('encoded', 'dir')
        mkdir('encoded');
    end
    
    % Process for each quantization level
    for m = m_bits
        encoded_file = sprintf('encoded/pan_tadeusz1_%dbit.enc', m);
        fid = fopen(encoded_file, 'wb');
        
        % Write header: sampling rate, bits per sample, number of segments
        num_segments = ceil((length(y)-r)/(N-r));
        fwrite(fid, Fs, 'uint32');
        fwrite(fid, m, 'uint8');
        fwrite(fid, num_segments, 'uint32');
        
        % Segment processing
        for seg = 1:num_segments
            % Get segment with overlap
            start_idx = max(1, (seg-1)*(N-r)+1);
            end_idx = min(length(y), start_idx+N-1);
            segment = y(start_idx:end_idx);
            
            % Ensure we have exactly N samples (pad with zeros if needed)
            if length(segment) < N
                segment = [segment, zeros(1, N-length(segment))];
            end
            
            % Apply window function (Hamming-like)
            w = 0.5*(1 - cos(2*pi*(0:N-1)/(N-1)));
            windowed_segment = segment .* w;
            
            % Pad with zeros (now ensuring consistent orientation)
            padded_segment = [zeros(1,r), windowed_segment, zeros(1,r)];
            
            % Levinson-Durbin algorithm to find AR coefficients
            [a, ~] = levinson_durbin(padded_segment, r);
            
            % Calculate residual errors
            errors = filter(a, 1, padded_segment);
            errors = errors(r+1:r+N); % Remove zero-padding effects
            
            % Find maximum error for scaling
            emax = max(abs(errors));
            if emax == 0, emax = 1; end % Avoid division by zero
            
            % Added: Adaptive quantization with triangular dither
            levels = 2^m;
            quant_step = emax / 8; % Initial step size
            quantized_errors = zeros(1, length(errors));
            for k = 1:length(errors)
                dither = quant_step * (rand() - rand()); % Triangular dither to reduce distortion
                delta = (errors(k) + dither) / quant_step;
                quantized_delta = round(delta * (levels-1)/2) / ((levels-1)/2);
                quantized_errors(k) = quantized_delta * quant_step;
                % Adapt step size based on signal magnitude
                quant_step = quant_step * (1.2 * abs(quantized_delta) + 0.8);
                quant_step = max(quant_step/10, min(quant_step, emax)); % Clamp
            end
            
            % Write to file
            fwrite(fid, a(2:end), 'float32'); % a(1) is always 1
            % Modified: Write initial quant_step instead of emax
            fwrite(fid, quant_step, 'float32');
            
            % Pack quantized errors into bits
            packed_errors = pack_errors(quantized_errors, emax, m);
            fwrite(fid, length(packed_errors), 'uint16');
            fwrite(fid, packed_errors, 'uint8');
        end
        
        fclose(fid);
        
        % Calculate compression ratio
        original_size = dir('pan_tadeusz1.wav').bytes;
        encoded_size = dir(encoded_file).bytes;
        compression_ratio = original_size / encoded_size;
        
        fprintf('m=%d bits: Compression ratio = %.2f:1\n', m, compression_ratio);
    end
end

% Levinson-Durbin (unchanged)
function [a, E] = levinson_durbin(x, p)
    % Autocorrelation
    N = length(x);
    r = zeros(1, p+1);
    for k = 0:p
        r(k+1) = x(1:N-k) * x(k+1:N)';
    end
    
    % Initialize
    a = zeros(1, p+1);
    a(1) = 1;
    E = r(1);
    
    % Levinson-Durbin recursion
    for k = 1:p
        lambda = -r(k+1:-1:2) * a(1:k)' / E;
        a(1:k+1) = [a(1:k), 0] + lambda * [0, fliplr(a(1:k))];
        E = E * (1 - lambda^2);
    end
end

% Pack Errors (unchanged)
function packed = pack_errors(errors, emax, m)
    % Normalize errors to 0..2^m-1 range
    levels = 2^m;
    normalized = round((errors + emax) * (levels-1) / (2*emax));
    normalized = max(0, min(normalized, levels-1)); % Clamp
    
    % Pack into bytes
    bits_per_byte = 8;
    num_errors = length(normalized);
    packed = zeros(1, ceil(num_errors * m / bits_per_byte), 'uint8');
    
    bit_pos = 0;
    byte_pos = 1;
    current_byte = 0;
    
    for i = 1:num_errors
        current_byte = bitor(current_byte, bitshift(uint8(normalized(i)), bit_pos));
        bit_pos = bit_pos + m;
        
        while bit_pos >= bits_per_byte
            packed(byte_pos) = bitand(current_byte, 255);
            byte_pos = byte_pos + 1;
            current_byte = bitshift(current_byte, -bits_per_byte);
            bit_pos = bit_pos - bits_per_byte;
        end
    end
    
    % Add remaining bits if any
    if bit_pos > 0
        packed(byte_pos) = current_byte;
    end
end

% Decoder
function predictive_decoder()
    % Get all encoded files
    encoded_files = dir('encoded/pan_tadeusz1_*bit.enc');
    
    % Read original audio for comparison
    [original_signal, original_Fs] = audioread('pan_tadeusz1.wav');
    if original_Fs ~= 11025
        original_signal = resample(original_signal, 11025, original_Fs);
    end
    if size(original_signal,2) > 1
        original_signal = mean(original_signal,2); % Convert to mono if stereo
    end
    original_signal = original_signal(:)'; % Ensure row vector
    
    % Added: Normalize original signal for fair comparison
    original_signal = original_signal / max(abs(original_signal));
    
    for i = 1:length(encoded_files)
        filename = encoded_files(i).name;
        fid = fopen(['encoded/' filename], 'rb');
        
        % Read header
        Fs = fread(fid, 1, 'uint32');
        m = fread(fid, 1, 'uint8');
        num_segments = fread(fid, 1, 'uint32');
        
        % Initialize output signal
        reconstructed = [];
        prev_samples = zeros(1, 10); % r=10
        
        % Process each segment
        for seg = 1:num_segments
            % Read AR coefficients
            a = [1, fread(fid, 10, 'float32')']; % a(1) is always 1
            
            % Modified: Read initial quant_step instead of emax
            quant_step = fread(fid, 1, 'float32');
            packed_length = fread(fid, 1, 'uint16');
            packed_errors = fread(fid, packed_length, 'uint8')';
            
            % Unpack errors
            errors = unpack_errors(packed_errors, m, 256);
            
            % Added: Reconstruct residuals with adaptive step
            levels = 2^m;
            quant_step_initial = quant_step; % Store initial step size
            reconstructed_errors = zeros(1, 256);
            for k = 1:256
                reconstructed_errors(k) = errors(k) * quant_step;
                quant_step = quant_step * (1.2 * abs(errors(k)) + 0.8);
                quant_step = max(quant_step_initial/10, min(quant_step, quant_step_initial));
            end
            errors = reconstructed_errors;
            
            % Reconstruct signal
            segment = zeros(1, 256);
            for k = 1:256
                if k <= 10
                    % Use previous segment's samples for first 10 samples
                    if seg == 1
                        input = [zeros(1,11-k), segment(1:k-1)];
                    else
                        input = [prev_samples(11-k:10), segment(1:k-1)];
                    end
                else
                    input = segment(k-10:k-1);
                end
                
                % Ensure exactly 10 samples
                if length(input) < 10
                    input = [zeros(1,10-length(input)), input];
                elseif length(input) > 10
                    input = input(end-9:end);
                end
                
                % Calculate prediction
                prediction = -sum(a(2:11) .* fliplr(input));
                segment(k) = prediction + errors(k);
            end
            
            % Update previous samples (with overlap)
            prev_samples = segment(end-9:end);
            
            % Modified: Overlap-add with Hann window to reduce discontinuities
            overlap = 10;
            hann_window = 0.5 * (1 - cos(2*pi*(0:overlap-1)/(overlap-1)));
            if isempty(reconstructed)
                reconstructed = segment;
            else
                overlap_start = length(reconstructed) - overlap + 1;
                overlap_end = length(reconstructed);
                reconstructed(overlap_start:overlap_end) = ...
                    reconstructed(overlap_start:overlap_end) .* (1-hann_window) + ...
                    segment(1:overlap) .* hann_window;
                reconstructed = [reconstructed, segment(overlap+1:end)];
            end
        end
        
        fclose(fid);
        
        % Added: Apply de-emphasis filter to reverse pre-emphasis
        pre_emphasis_coeff = 0.95;
        reconstructed = filter(1, [1, -pre_emphasis_coeff], reconstructed);
        
        % Trim reconstructed signal to match original length
        reconstructed = reconstructed(1:min(length(reconstructed), length(original_signal)));
        original_signal = original_signal(1:length(reconstructed));
        
        % Calculate MSE and SNR
        error_signal = original_signal - reconstructed;
        mse = mean(error_signal.^2);
        
        signal_power = mean(original_signal.^2);
        noise_power = mse;
        snr_db = 10*log10(signal_power/noise_power);
        
        % Save reconstructed WAV
        output_file = strrep(filename, '.enc', '.wav');
        if ~exist('decoded', 'dir')
            mkdir('decoded');
        end
        audiowrite(['decoded/' output_file], reconstructed, Fs);
        
        % Display results
        fprintf('\nResults for %d-bit quantization:\n', m);
        fprintf('Compression ratio: %.2f:1\n', dir('pan_tadeusz1.wav').bytes/dir(['encoded/' filename]).bytes);
        fprintf('MSE: %.6f\n', mse);
        fprintf('SNR: %.2f dB\n', snr_db);
        fprintf('Decoded file saved as: decoded/%s\n\n', output_file);
    end
end

% Unpack Errors
function errors = unpack_errors(packed, m, N)
    % Unpack bits to errors
    levels = 2^m;
    bits_per_byte = 8;
    errors = zeros(1, N);
    
    bit_pos = 0;
    byte_pos = 1;
    current_byte = packed(1);
    
    for i = 1:N
        % Extract m bits
        mask = bitshift(1, m) - 1;
        value = bitand(bitshift(current_byte, -bit_pos), mask);
        
        remaining_bits = bits_per_byte - bit_pos;
        if remaining_bits < m
            byte_pos = byte_pos + 1;
            if byte_pos <= length(packed)
                next_byte = packed(byte_pos);
                value = bitor(value, bitshift(bitand(next_byte, bitshift(1, m - remaining_bits) - 1), remaining_bits));
                current_byte = next_byte;
                bit_pos = m - remaining_bits;
            else
                bit_pos = bit_pos + m;
            end
        else
            bit_pos = bit_pos + m;
            if bit_pos >= bits_per_byte
                byte_pos = byte_pos + 1;
                if byte_pos <= length(packed)
                    current_byte = packed(byte_pos);
                end
                bit_pos = bit_pos - bits_per_byte;
            end
        end
        
        % Modified: Convert back to error value, assuming quant_step scaling in decoder
        errors(i) = (value * 2 / (levels-1) - 1); % Scale to [-1, 1] for adaptive quantization
    end
end