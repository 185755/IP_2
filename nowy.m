% audio_compression_stable.m
% Implementuje kodowanie predykcyjne z optymalną kompresją
% Generuje pełne pliki audio zrekonstruowanego sygnału z poprawnym dodawaniem nakładającym się
% Odtwarza cały zrekonstruowany klip
% Opcjonalnie zapisuje pliki segmentów do debugowania

% Parametry
fs = 11025;         % Częstotliwość próbkowania
N = 256;           % Rozmiar segmentu (zgodnie z dokumentem)
r = 10;            % Rząd modelu AR
overlap = 20;      % Próbki nakładania
quant_bits = [2, 3, 4]; % Poziomy kwantyzacji
epsilon = 1e-6;    % Mała stała dla okna
residual_scale = 1000; % Skala reszt do uniknięcia utraty kwantyzacji
save_segments = true; % Flaga do zapisu plików segmentów debugowania

% Odczyt pliku WAV wejściowego (zakładając 16-bitowy PCM, mono)
try
    [audio, fs_in] = audioread('pan_tadeusz1.wav');
catch e
    error('Nie udało się odczytać pan_tadeusz1.wav: %s', e.message);
end
assert(fs_in == fs, 'Częstotliwość próbkowania musi wynosić 11025 Hz');
assert(size(audio,2) == 1, 'Wejście musi być mono');
audio = audio(:)'; % Upewnij się, że jest wektorem wierszowym

% Sprawdzenie amplitudy i czasu trwania sygnału wejściowego
input_amplitude = max(abs(audio));
input_duration = length(audio) / fs;
if input_amplitude < 1e-6
    warning('Amplituda sygnału wejściowego jest bardzo niska (max = %.2e). Wyjście może być ciche.', input_amplitude);
end

%% Funkcja okna (Hanning dla "spłaszczenia" na krawędziach)
window = 0.5 * [1 - cos(2 * pi * (0:N-1) / (N+1))]; % Okno Hanninga zgodnie z dokumentem

% Inicjalizacja pliku wyjściowego dla raportu rekonstrukcji
fid_report = fopen('reconstruction_report.txt', 'w');
fprintf(fid_report, 'Raport Rekonstrukcji\n');
fprintf(fid_report, '====================\n');

% Inicjalizacja danych wykresu
residual_amplitudes = [];
recon_amplitudes = [];

for m = quant_bits
    % Nadajnik: Kompresja
    num_samples = length(audio);
    num_segments = floor((num_samples - N) / (N - overlap)) + 1;
    
    % Inicjalizacja pamięci dla danych skompresowanych
    all_a = zeros(num_segments, r, 'single'); % float32 dla współczynników
    all_emax = zeros(num_segments, 1, 'single'); % float32 dla e_max
    all_e_quant_indices = zeros(num_segments, N, 'uint8'); % Indeksy kwantyzacji
    
    % Przetwarzanie każdego segmentu
    for seg = 1:num_segments
        %% Wyodrębnienie segmentu z nakładaniem
        start_idx = 1 + (seg-1) * (N - overlap);
        end_idx = min(start_idx + N - 1, num_samples);
        segment = audio(start_idx:end_idx);
        if length(segment) < N
            segment = [segment, zeros(1, N - length(segment))];
        end
        
        %% Zastosowanie okna dla "spłaszczenia"
        segment_windowed = segment .* window;
        
        %% Wypełnienie zerami po obydwu stronach
        padded_segment = [zeros(1, r), segment_windowed, zeros(1, r)];
        
        %% Algorytm Levinson-Durbin dla współczynników AR
        [a, k] = levinson_durbin(padded_segment, r);
        
        % Sprawdzenie stabilności AR za pomocą współczynników odbicia
        if any(abs(k) >= 1)
            fprintf(fid_report, 'Ostrzeżenie: Niestabilne współczynniki AR w segmencie %d (m=%d bitów). Dostosowanie.\n', seg, m);
            k = min(max(k, -0.99), 0.99); % Przycięcie współczynników odbicia
            a = k2a(k); % Konwersja z powrotem na współczynniki AR
        end
        
        % Zapis współczynników AR
        all_a(seg, :) = a(2:end); % Wyklucz a(0)=1
        
        %% Obliczenie błędów resztowych
        a_full = [1, a(2:end)];
        e = filter(a_full, 1, segment_windowed);
        e = e(1:N); % Przycięcie do długości segmentu
        
        % Sprawdzenie amplitudy resztkowej
        e_max = max(abs(e));
        all_emax(seg) = e_max;
        if e_max < 1e-6
            fprintf(fid_report, 'Ostrzeżenie: Segment %d (m=%d bitów) ma bardzo niską e_max (%.2e).\n', seg, m, e_max);
            e_max = 1e-6; % Zapobieganie dzieleniu przez zero
        end
        
        %% Kwantyzacja błędów resztowych
        e_scaled = e * residual_scale;
        e_max_scaled = e_max * residual_scale;
        levels = 2^m;
        delta = 2 * e_max_scaled / (levels - 1);
        e_quant_indices = round((e_scaled + e_max_scaled) / delta); % Mapowanie na [0, 2^m-1]
        e_quant_indices = max(0, min(levels-1, e_quant_indices)); % Przycięcie
        all_e_quant_indices(seg, :) = e_quant_indices;
        
        residual_amplitudes = [residual_amplitudes, e_max];
    end
    
    %% Zapis danych do pliku binarnego
    filename = sprintf('compressed_m%d.bin', m);
    fid = fopen(filename, 'w');
    fwrite(fid, all_a(:), 'float32'); % 32 bity na współczynnik
    fwrite(fid, all_emax, 'float32'); % 32 bity na e_max
    fwrite(fid, all_e_quant_indices(:), 'uint8'); % 8 bitów na indeks (wypełnione)
    fwrite(fid, residual_scale, 'float32'); % 32 bity
    fwrite(fid, m, 'uint8'); % Zapis bitów kwantyzacji
    fclose(fid);
    
    % Obliczenie stosunku kompresji
    bits_per_residual = m; % Rzeczywiste bity użyte w kwantyzacji
    compressed_size = (num_segments * r * 32 + num_segments * 32 + num_segments * N * 8 + 32 + 8); % bity
    original_size = num_samples * 16; % bity
    compression_ratio = original_size / compressed_size;
    
    % Pobranie rzeczywistego rozmiaru pliku
    file_info = dir(filename);
    compressed_bytes = file_info.bytes;
    
    % Odbiornik: Dekompresja
    reconstructed = zeros(1, num_samples);
    weight_sum = zeros(1, num_samples); % Śledzenie wag nakładania
    overlap_window = ones(1, N); % Jednolite okno dla dodawania nakładającego się
    
    %% Odczyt danych z pliku binarnego
    fid = fopen(filename, 'r');
    read_a = fread(fid, [num_segments, r], 'float32');
    read_emax = fread(fid, num_segments, 'float32');
    read_e_quant_indices = fread(fid, [num_segments, N], 'uint8');
    read_scale = fread(fid, 1, 'float32');
    read_m = fread(fid, 1, 'uint8');
    fclose(fid);
    
    %% Rekonstrukcja sygnału z dodawaniem nakładającym się
    for seg = 1:num_segments
        a = [1, read_a(seg, :)]; % Włącz a(0)=1
        e_max = read_emax(seg);
        e_quant_indices = read_e_quant_indices(seg, :);
        
        start_idx = 1 + (seg-1) * (N - overlap);
        end_idx = min(start_idx + N - 1, num_samples);
        
        if e_max < 1e-6
            segment_recon = zeros(1, N);
        else
            levels = 2^m;
            delta = 2 * (e_max * read_scale) / (levels - 1);
            e_quant = (e_quant_indices * delta - (e_max * read_scale)) / read_scale;
            segment_recon = filter(1, a, e_quant);
            segment_recon = segment_recon(1:N);
        end
        
        if any(isnan(segment_recon)) || any(isinf(segment_recon))
            fprintf(fid_report, 'Ostrzeżenie: NaN lub Inf w segmencie %d (m=%d bitów). Ustawienie na zero.\n', seg, m);
            segment_recon(isnan(segment_recon) | isinf(segment_recon)) = 0;
        end
        
        recon_amplitudes = [recon_amplitudes, max(abs(segment_recon))];
        
        if end_idx >= start_idx
            segment_len = end_idx - start_idx + 1;
            reconstructed(start_idx:end_idx) = reconstructed(start_idx:end_idx) + segment_recon(1:segment_len) .* overlap_window(1:segment_len);
            weight_sum(start_idx:end_idx) = weight_sum(start_idx:end_idx) + overlap_window(1:segment_len);
        end
    end
    
    % Normalizacja regionów nakładających się
    weight_sum(weight_sum == 0) = 1; % Unikanie dzielenia przez zero
    reconstructed = reconstructed ./ weight_sum;
    
    % Weryfikacja pełnego sygnału zrekonstruowanego
    recon_amplitude = max(abs(reconstructed));
    recon_duration = length(reconstructed) / fs;
    if recon_amplitude < 1e-6
        fprintf(fid_report, 'Ostrzeżenie: Pełny sygnał zrekonstruowany (m=%d bitów) ma prawie zerową amplitudę (%.2e).\n', m, recon_amplitude);
    end
    
    % Normalizacja do dopasowania amplitudy wejściowej
    if recon_amplitude > 0
        reconstructed = reconstructed * (input_amplitude / recon_amplitude);
    end
    
    %% Zapis zrekonstruowanego sygnału do pliku WAV
    recon_filename = sprintf('reconstructed_m%d.wav', m);
    audiowrite(recon_filename, reconstructed, fs);
    fprintf(fid_report, 'Zapisano pełny plik zrekonstruowany: %s\n', recon_filename);
    fprintf(fid_report, 'Full reconstructed max amplitude: %.2e\n', max(abs(reconstructed)));
    fprintf(fid_report, 'Czas trwania pełnego zrekonstruowanego: %.2f sekund\n', recon_duration);
    fprintf(fid_report, 'Rozmiar pliku skompresowanego: %d bajtów\n', compressed_bytes);
    
    % Odtwarzanie całego zrekonstruowanego klipu
    fprintf('Odtwarzanie pełnego zrekonstruowanego klipu dla m=%d bitów (%.2f sekund)...\n', m, recon_duration);
    soundsc(reconstructed, fs);
    pause(recon_duration + 1); % Poczekaj na zakończenie odtwarzania plus 1-sekundowy bufor
    
    % Obliczenie SNR
    noise = audio(1:length(reconstructed)) - reconstructed;
    signal_power = mean(audio(1:length(reconstructed)).^2);
    noise_power = mean(noise.^2);
    if noise_power > 0
        snr_db = 10 * log10(signal_power / noise_power);
    else
        snr_db = Inf;
        fprintf(fid_report, 'Ostrzeżenie: Zerowa moc szumu dla m=%d bitów. SNR ustawione na Inf.\n', m);
    end
    
    %% Zapis stopnia kompresji do raportu
    fprintf(fid_report, 'Kwantyzacja: %d bitów\n', m);
    fprintf(fid_report, 'Stopień kompresji: %.2f\n', compression_ratio);
    fprintf(fid_report, 'SNR: %.2f dB\n', snr_db);
    fprintf(fid_report, '\n');
    
    % Wykres amplitud resztkowych i zrekonstruowanych
    figure;
    subplot(2,1,1);
    plot(1:length(residual_amplitudes), residual_amplitudes, 'b');
    title(sprintf('Amplitudy Resztkowe (m=%d bitów)', m));
    xlabel('Segment');
    ylabel('Maksymalna Amplituda');
    subplot(2,1,2);
    plot(1:length(recon_amplitudes), recon_amplitudes, 'r');
    title(sprintf('Amplitudy Zrekonstruowane (m=%d bitów)', m));
    xlabel('Segment');
    ylabel('Maksymalna Amplituda');
    saveas(gcf, sprintf('amplitudes_m%d.png', m));
    
    residual_amplitudes = [];
    recon_amplitudes = [];
end

fclose(fid_report);

%% Implementacja Levinson-Durbin
function [a, k] = levinson_durbin(x, p)
    N = length(x);
    % Autokorelacja
    R = zeros(1, p+1);
    for m = 0:p
        sum_val = 0;
        for n = 1:N-m
            sum_val = sum_val + x(n) * x(n+m);
        end
        R(m+1) = sum_val;
    end
    
    % Inicjalizacja
    a = zeros(1, p+1);
    a(1) = 1;
    E = R(1);
    k = zeros(1, p);
    
    % Rekursja Levinson-Durbin
    for m = 1:p
        lambda = 0;
        for j = 1:m
            lambda = lambda + a(j) * R(m-j+2);
        end
        if E == 0
            k(m) = 0; % Zapobieganie dzieleniu przez zero
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

%% Konwersja współczynników odbicia na współczynniki AR
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