% audio_compression_stable.m
% Implements predictive coding with optimized compression
% Generates full reconstructed audio files with proper overlap-add
% Plays entire reconstructed clip
% Optionally saves segment files for debugging

% Parameters
fs = 11025;         % Sampling frequency
N = 512;           % Segment size (increased for fewer segments)
r = 10;            % AR model order
overlap = 20;      % Overlap samples (increased)
quant_bits = [2, 3, 4]; % Quantization levels
epsilon = 1e-6;    % Small constant for window
residual_scale = 1000; % Scale residuals to avoid quantization loss
save_segments = true; % Flag to save debug segment files

% Read input WAV file (assuming 16-bit PCM, mono)
try
    [audio, fs_in] = audioread('pan_tadeusz1.wav');
catch e
    error('Failed to read pan_tadeusz1.wav: %s', e.message);
end
assert(fs_in == fs, 'Sampling frequency must be 11025 Hz');
assert(size(audio,2) == 1, 'Input must be mono');
audio = audio(:)'; % Ensure row vector

% Check input signal amplitude and duration
input_amplitude = max(abs(audio));
input_duration = length(audio) / fs;
if input_amplitude < 1e-6
    warning('Input signal amplitude is very low (max = %.2e). Output may be silent.', input_amplitude);
end

% Window function (rectangular to avoid inverse issues)
window = ones(1, N);

% Initialize output file for reconstruction report
fid_report = fopen('reconstruction_report.txt', 'w');
fprintf(fid_report, 'Reconstruction Report\n');
fprintf(fid_report, '====================\n');
fprintf(fid_report, 'Input file: pan_tadeusz1.wav\n');
fprintf(fid_report, 'Sampling frequency: %d Hz\n', fs);
fprintf(fid_report, 'Segment size: %d samples\n', N);
fprintf(fid_report, 'AR model order: %d\n', r);
fprintf(fid_report, 'Overlap: %d samples\n');
fprintf(fid_report, 'Input signal max amplitude: %.2e\n', input_amplitude);
fprintf(fid_report, 'Input duration: %.2f seconds\n');
fprintf(fid_report, 'Input size: %d bytes\n\n', numel(audio) * 2); % 16-bit = 2 bytes per sample

% Output first 10 samples of input signal
fprintf(fid_report, 'First 10 samples of input signal:\n');
fprintf(fid_report, '%.6f\n', audio(1:min(10, length(audio))));

% Initialize plot data
residual_amplitudes = [];
recon_amplitudes = [];

for m = quant_bits
    % Transmitter: Compression
    num_samples = length(audio);
    num_segments = floor((num_samples - N) / (N - overlap)) + 1;
    
    % Initialize storage for compressed data
    all_a = zeros(num_segments, r, 'single'); % float32 for coefficients
    all_emax = zeros(num_segments, 1, 'single'); % float32 for e_max
    all_e_quant_indices = zeros(num_segments, N, 'uint8'); % Quantization indices
    
    % Process each segment
    for seg = 1:num_segments
        % Extract segment with overlap
        start_idx = 1 + (seg-1) * (N - overlap);
        end_idx = min(start_idx + N - 1, num_samples);
        segment = audio(start_idx:end_idx);
        if length(segment) < N
            segment = [segment, zeros(1, N - length(segment))];
        end
        
        % Apply window
        segment_windowed = segment .* window;
        
        % Pad with zeros
        padded_segment = [zeros(1, r), segment_windowed, zeros(1, r)];
        
        % Levinson-Durbin algorithm for AR coefficients
        [a, k] = levinson_durbin(padded_segment, r);
        
        % Check AR stability via reflection coefficients
        if any(abs(k) >= 1)
            fprintf(fid_report, 'Warning: Unstable AR coefficients in segment %d (m=%d bits). Adjusting.\n', seg, m);
            k = min(max(k, -0.99), 0.99); % Clamp reflection coefficients
            a = k2a(k); % Convert back to AR coefficients
        end
        
        % Store AR coefficients
        all_a(seg, :) = a(2:end); % Exclude a(0)=1
        
        % Calculate residual error using filter
        a_full = [1, a(2:end)];
        e = filter(a_full, 1, segment_windowed);
        e = e(1:N); % Trim to segment length
        
        % Check residual amplitude
        e_max = max(abs(e));
        all_emax(seg) = e_max;
        if e_max < 1e-6
            fprintf(fid_report, 'Warning: Segment %d (m=%d bits) has very low e_max (%.2e).\n', seg, m, e_max);
            e_max = 1e-6; % Prevent division by zero
        end
        
        % Scale residuals to avoid quantization loss
        e_scaled = e * residual_scale;
        e_max_scaled = e_max * residual_scale;
        
        % Quantization
        levels = 2^m;
        delta = 2 * e_max_scaled / (levels - 1);
        e_quant_indices = round((e_scaled + e_max_scaled) / delta); % Map to [0, 2^m-1]
        e_quant_indices = max(0, min(levels-1, e_quant_indices)); % Clamp
        all_e_quant_indices(seg, :) = e_quant_indices;
        
        % Log residual amplitude
        fprintf(fid_report, 'Segment %d (m=%d bits) max residual amplitude: %.2e\n', seg, m, e_max);
        residual_amplitudes = [residual_amplitudes, e_max];
    end
    
    % Save compressed data
    filename = sprintf('compressed_m%d.bin', m);
    fid = fopen(filename, 'w');
    fwrite(fid, all_a(:), 'float32'); % 32 bits per coefficient
    fwrite(fid, all_emax, 'float32'); % 32 bits per e_max
    fwrite(fid, all_e_quant_indices(:), 'uint8'); % 8 bits per index (padded)
    fwrite(fid, residual_scale, 'float32'); % 32 bits
    fwrite(fid, m, 'uint8'); % Store quantization bits
    fclose(fid);
    
    % Calculate compression ratio
    bits_per_residual = m; % Actual bits used in quantization
    compressed_size = (num_segments * r * 32 + num_segments * 32 + num_segments * N * 8 + 32 + 8); % bits
    original_size = num_samples * 16; % bits
    compression_ratio = original_size / compressed_size;
    
    % Get actual file size
    file_info = dir(filename);
    compressed_bytes = file_info.bytes;
    
    % Receiver: Decompression
    reconstructed = zeros(1, num_samples);
    weight_sum = zeros(1, num_samples); % Track overlap weights
    overlap_window = ones(1, N); % Uniform weight for overlap-add
    
    % Read compressed data
    fid = fopen(filename, 'r');
    read_a = fread(fid, [num_segments, r], 'float32');
    read_emax = fread(fid, num_segments, 'float32');
    read_e_quant_indices = fread(fid, [num_segments, N], 'uint8');
    read_scale = fread(fid, 1, 'float32');
    read_m = fread(fid, 1, 'uint8');
    fclose(fid);
    
    % Check for file read errors
    if any(size(read_a) ~= [num_segments, r]) || length(read_emax) ~= num_segments || any(size(read_e_quant_indices) ~= [num_segments, N]) || isempty(read_scale) || isempty(read_m)
        fprintf(fid_report, 'Error: File read error for m=%d bits. Check compressed file integrity.\n', m);
        continue;
    end
    assert(read_m == m, 'Quantization bits mismatch in compressed file.');
    
    % Reconstruct signal with overlap-add
    for seg = 1:num_segments
        a = [1, read_a(seg, :)]; % Include a(0)=1
        e_max = read_emax(seg);
        e_quant_indices = read_e_quant_indices(seg, :);
        
        start_idx = 1 + (seg-1) * (N - overlap);
        end_idx = min(start_idx + N - 1, num_samples);
        
        % Skip silent segments
        if e_max < 1e-6
            segment_recon = zeros(1, N);
        else
            % Reconstruct residuals
            levels = 2^m;
            delta = 2 * (e_max * read_scale) / (levels - 1);
            e_quant = (e_quant_indices * delta - (e_max * read_scale)) / read_scale;
            % Reconstruct using filter
            segment_recon = filter(1, a, e_quant);
            segment_recon = segment_recon(1:N);
        end
        
        % Check for NaNs or Infs
        if any(isnan(segment_recon)) || any(isinf(segment_recon))
            fprintf(fid_report, 'Warning: NaN or Inf in segment %d (m=%d bits). Setting to zero.\n', seg, m);
            segment_recon(isnan(segment_recon) | isinf(segment_recon)) = 0;
        end
        
        % Log segment amplitude
        fprintf(fid_report, 'Segment %d (m=%d bits) max reconstructed amplitude: %.2e\n', seg, m, max(abs(segment_recon)));
        recon_amplitudes = [recon_amplitudes, max(abs(segment_recon))];
        
        % Overlap-add
        if end_idx >= start_idx
            segment_len = end_idx - start_idx + 1;
            reconstructed(start_idx:end_idx) = reconstructed(start_idx:end_idx) + segment_recon(1:segment_len) .* overlap_window(1:segment_len);
            weight_sum(start_idx:end_idx) = weight_sum(start_idx:end_idx) + overlap_window(1:segment_len);
        end
        
        % Save segment for debugging (optional, first 20 segments)
        if save_segments && seg <= 20
            audiowrite(sprintf('debug_segment_%d_m%d.wav', seg, m), segment_recon, fs);
        end
    end
    
    % Normalize overlapping regions
    weight_sum(weight_sum == 0) = 1; % Avoid division by zero
    reconstructed = reconstructed ./ weight_sum;
    
    % Handle remaining samples
    if end_idx < num_samples
        reconstructed(end_idx+1:num_samples) = audio(end_idx+1:num_samples); % Pad with original audio
    end
    
    % Verify full reconstructed signal
    recon_amplitude = max(abs(reconstructed));
    recon_duration = length(reconstructed) / fs;
    if recon_amplitude < 1e-6
        fprintf(fid_report, 'Warning: Full reconstructed signal (m=%d bits) has near-zero amplitude (%.2e).\n', m, recon_amplitude);
    end
    
    % Normalize to match input amplitude
    if recon_amplitude > 0
        reconstructed = reconstructed * (input_amplitude / recon_amplitude);
    end
    
    % Save full reconstructed audio
    recon_filename = sprintf('reconstructed_m%d.wav', m);
    audiowrite(recon_filename, reconstructed, fs);
    fprintf(fid_report, 'Saved full reconstructed file: %s\n', recon_filename);
    fprintf(fid_report, 'Full reconstructed max amplitude: %.2e\n', max(abs(reconstructed)));
    fprintf(fid_report, 'Full reconstructed duration: %.2f seconds\n', recon_duration);
    fprintf(fid_report, 'Compressed file size: %d bytes\n', compressed_bytes);
    
    % Play entire reconstructed clip
    fprintf('Playing full reconstructed clip for m=%d bits (%.2f seconds)...\n', m, recon_duration);
    soundsc(reconstructed, fs);
    pause(recon_duration + 1); % Wait for playback to finish plus 1 second buffer
    
    % Calculate SNR
    noise = audio(1:length(reconstructed)) - reconstructed;
    signal_power = mean(audio(1:length(reconstructed)).^2);
    noise_power = mean(noise.^2);
    if noise_power > 0
        snr_db = 10 * log10(signal_power / noise_power);
    else
        snr_db = Inf;
        fprintf(fid_report, 'Warning: Zero noise power for m=%d bits. SNR set to Inf.\n', m);
    end
    
    % Output first 10 samples of reconstructed signal
    fprintf(fid_report, 'First 10 samples of reconstructed signal (m=%d bits):\n', m);
    fprintf(fid_report, '%.6f\n', reconstructed(1:min(10, length(reconstructed))));
    
    % Write to report
    fprintf(fid_report, 'Quantization: %d bits\n', m);
    fprintf(fid_report, 'Compression ratio: %.2f\n', compression_ratio);
    fprintf(fid_report, 'SNR: %.2f dB\n', snr_db);
    fprintf(fid_report, '\n');
    
    % Plot residual and reconstructed amplitudes
    figure;
    subplot(2,1,1);
    plot(1:length(residual_amplitudes), residual_amplitudes, 'b');
    title(sprintf('Residual Amplitudes (m=%d bits)', m));
    xlabel('Segment');
    ylabel('Max Amplitude');
    subplot(2,1,2);
    plot(1:length(recon_amplitudes), recon_amplitudes, 'r');
    title(sprintf('Reconstructed Amplitudes (m=%d bits)', m));
    xlabel('Segment');
    ylabel('Max Amplitude');
    saveas(gcf, sprintf('amplitudes_m%d.png', m));
    
    % Reset plot data for next m
    residual_amplitudes = [];
    recon_amplitudes = [];
end

fclose(fid_report);

% Levinson-Durbin implementation with reflection coefficients
function [a, k] = levinson_durbin(x, p)
    N = length(x);
    % Autocorrelation
    R = zeros(1, p+1);
    for m = 0:p
        sum_val = 0;
        for n = 1:N-m
            sum_val = sum_val + x(n) * x(n+m);
        end
        R(m+1) = sum_val;
    end
    
    % Initialize
    a = zeros(1, p+1);
    a(1) = 1;
    E = R(1);
    k = zeros(1, p);
    
    % Levinson-Durbin recursion
    for m = 1:p
        lambda = 0;
        for j = 1:m
            lambda = lambda + a(j) * R(m-j+2);
        end
        if E == 0
            k(m) = 0; % Prevent division by zero
        else
            k(m) = -lambda / E;
        end
        a_new = a;
        for j = 1:m
            a_new(j+1) = a(j+1) + k(m) * a(m-j+1);
        end
        a = a_new;
        E = E * (1 - k(m)^2);
    end
end

% Convert reflection coefficients to AR coefficients
function a = k2a(k)
    p = length(k);
    a = [1, zeros(1, p)];
    for m = 1:p
        a_new = a;
        for j = 1:m
            a_new(j+1) = a(j+1) + k(m) * a(m-j+1);
        end
        a = a_new;
    end
end