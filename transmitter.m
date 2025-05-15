clear all;
clc;

%% Wczytanie i przygotowanie sygnału wejściowego
[x1, fs] = audioread("pan_tadeusz1.wav");
x1 = x1(:,1); % wybór tylko jednego kanału (mono)

%% Parametry kodowania
N = 256;     % długość segmentu
r = 10;      % rząd modelu AR
m = 3;       % liczba bitów dla kwantyzacji (zmień na 2, 3, 4 według potrzeby)

%% Podział sygnału na segmenty z nakładaniem, spłaszczenie i zerowanie końców
podzielone = podzial(x1);                   % segmentacja sygnału (nakładanie 10 próbek)
splaszczone = splaszcz(podzielone);         % zastosowanie okna wygładzającego (Hanning)
wyzerowane_konce = wyzeruj_konce(splaszczone);  % dodanie zer z obu stron segmentu
num_segmentow = size(wyzerowane_konce, 1);  % liczba segmentów

%% Bufory na dane wynikowe
wszystkie_a = zeros(num_segmentow, r);      % współczynniki AR dla każdego segmentu
wszystkie_ek = zeros(num_segmentow, N);     % błędy resztowe po kwantyzacji
wszystkie_emax = zeros(num_segmentow, 1);   % maksymalna wartość bezwzględna błędu resztowego

%% Kodowanie każdego segmentu
for i = 1:num_segmentow
    segment = wyzerowane_konce(i,:);
    
    % Oblicz korelacje i współczynniki AR za pomocą algorytmu Levinsona-Durbina
    [pi, ~] = policz_pi_Ri(segment, r);
    [~, a_wspolczynniki, ~] = L_D(pi);
    a = a_wspolczynniki(2:end); % pomijamy 1 na początku
    
    % Wyodrębnienie użytecznego fragmentu (bez zer)
    y = segment(r+1:end-r);
    e = zeros(1, N); % błędy resztowe

    % Obliczenie błędów resztowych e(k) = y(k) + suma(a_i * y(k-i))
    for k = r+1 : r+N
        y_k = segment(k);
        predykcja = 0;
        for j = 1:r
            predykcja = predykcja - a(j) * segment(k-j);
        end
        e(k - r) = y_k + predykcja;
    end

    % Kwantyzacja błędów resztowych
    emax = max(abs(e));
    e_kwant = round((e + emax) * ((2^m - 1) / (2 * emax)));
    e_kwant = min(max(e_kwant, 0), 2^m - 1); % zabezpieczenie

    % Zapamiętaj dane dla segmentu
    wszystkie_a(i,:) = a;
    wszystkie_ek(i,:) = e_kwant;
    wszystkie_emax(i) = emax;
end

%% Zapis danych do pliku binarnego
plik = fopen("zakodowane_dane.bin", "w");

for i = 1:num_segmentow
    fwrite(plik, wszystkie_a(i,:), 'float32');     % współczynniki AR
    fwrite(plik, wszystkie_emax(i), 'float32');    % emax
    
    % Zamiana kwantowanych e(k) na strumień bitów i zapis jako bajty
    e_kwant = wszystkie_ek(i,:);
    bitstream = my_de2bi(e_kwant, m)';     % kolumnowy układ bitów
    bitstream = bitstream(:)';                     % konwersja do wektora 1D
    padding = mod(-length(bitstream),8);           % dopełnienie do pełnych bajtów
    bitstream = [bitstream, zeros(1,padding)];
    bytes = uint8(reshape(bitstream, 8, []).');     % grupowanie po 8 bitów
    fwrite(plik, bytes, 'uint8');                  % zapis jako bajty
end

fclose(plik);

% --- Odczyt danych z pliku ---
plik = fopen("zakodowane_dane.bin", "r");
dane = fread(plik, 'uint8');
fclose(plik);

ptr = 1;
a_all = {};
e_all = {};
emax_all = [];
i = 1;

while ptr + 4*r + 4 - 1 <= length(dane)
    a = typecast(uint8(dane(ptr:ptr+4*r-1)), 'single');
    ptr = ptr + 4*r;
    emax = typecast(uint8(dane(ptr:ptr+3)), 'single');
    ptr = ptr + 4;
    bit_count = m * N;
    byte_count = ceil(bit_count / 8);
    if ptr + byte_count - 1 > length(dane)
        break;
    end
    bytes = dane(ptr:ptr + byte_count - 1);
    ptr = ptr + byte_count;
    bitstream = reshape(dec2bin(bytes, 8).' - '0', 1, []);
    bitstream = bitstream(1 : m * N);
    e_kwant = reshape(bitstream, m, []).';
    e_kwant = my_bi2de(e_kwant, 'left-msb');
    e = e_kwant * (2 * emax / (2^m - 1)) - emax;
    a_all{i} = a;
    e_all{i} = e;
    emax_all(i) = emax;
    i = i + 1;
end

%% --- Rekonstrukcja sygnału ---
y_prev = zeros(1, r);
reconstructed = [];

for i = 1:length(e_all)
    a = a_all{i};
    e = e_all{i};
    y = zeros(1, N + r);
    y(1:r) = y_prev;
    for k = r+1 : N+r
        suma = 0;
        for j = 1:r
            suma = suma - a(j) * y(k-j);
        end
        y(k) = e(k-r) + suma;
    end
    reconstructed = [reconstructed, y(r+1:end)];
    y_prev = y(end - r + 1:end);
end

% --- Ocena jakości rekonstrukcji ---
original = x1; % oryginalny sygnał wczytany wcześniej
min_len = min(length(original), length(reconstructed));
original = original(1:min_len);
reconstructed = reconstructed(1:min_len);

original = original(:);      % zmienia na wektor kolumnowy
reconstructed = reconstructed(:); % zmienia na wektor kolumnowy
reconstructed = min(max(reconstructed, -1e3), 1e3); % ograniczenie wielkosci


mse = mean((original - reconstructed).^2);
snr_val = snr(reconstructed, original - reconstructed);

fprintf("\n=== Ocena jakości rekonstrukcji sygnału ===\n");
fprintf("Średni błąd kwadratowy (MSE): %.4f\n", mse);
fprintf("Stosunek sygnału do szumu (SNR): %.2f dB\n", snr_val);

%% Oblicz kompresję
info = dir("pan_tadeusz1.wav");
oryginalny_rozmiar = info.bytes;
info = dir("zakodowane_dane.bin");
skompresowany_rozmiar = info.bytes;

kompresja = oryginalny_rozmiar / skompresowany_rozmiar;
fprintf("Stopień kompresji: %.2f\n", kompresja);

%% --- Pomocnicze funkcje --- (niezmienione)

function podzielone = podzial(niepodzielone)
    podzielone = zeros(896,256);
    podzielone(1,:) = niepodzielone(1:256);
    start = 247;
    for i = 2 : 896
        podzielone(i,:) = niepodzielone(start: start+255).';
        start = start + 246;
    end
end

function splaszczone = splaszcz(niesplaszczone)
    wagi = zeros(1,256);
    splaszczone = zeros(size(niesplaszczone));
    for i = 1:256
        waga = 0.5*(1-cos(2*pi()/(256+1)*i));
        wagi(i)=waga;
    end
    for i = 1:896
        splaszczone(i,:) = niesplaszczone(i,:).*wagi;
    end
end

function wyzerowane_konce = wyzeruj_konce(niewyzerowane_konce)
    wymiar = size(niewyzerowane_konce);
    wyzerowane_konce = zeros(wymiar(1), 10+wymiar(2)+10);
    wyzerowane_konce(:,11:11+255) = niewyzerowane_konce;
end

function [pi,Ri] = policz_pi_Ri(y, r)
    suma = zeros(1, r);
    p0 = 0;
    N = size(y);
    for i = 1 : r
        for t = i+1 : N(2)
            suma(i) = suma(i) + y(t)*y(t-i);
        end
    end    
    for t = 1 : N(2)
            p0 = p0 + y(t)*y(t);
    end
    pi = [p0 suma];
    Ri = pi/N(2);
end

function [params,a, e] = L_D(pi)
    a = zeros(10);
    sigma = zeros(10,1);
    k = zeros(10,1);
    k1 = pi(2)/pi(1);
    k(1) = k1;
    sigma1 = (1-k1^2)*pi(1);
    sigma(1) = sigma1;
    a(1,1) = k1;
    for i = 2:10
        k(i) = (pi(i+1) - suma_ap(i,a,pi))/sigma(i-1);
        a(i,i) = k(i);
        for j = 1:i-1
            a(j,i) = a(j,i-1) - k(i)*a(i-j,i-1);
        end
        sigma(i) = (1-k(i)^2)*sigma(i-1);
    end
    params = k;
    a =[1 a(:,10)'];
    e = sigma(end);
end

function sumaAP = suma_ap(i,a,pi)
    sumaAP = 0;
    for j = 1:i-1
        sumaAP = sumaAP + a(j,i-1)*pi(i-j+1);
    end
end

function B = my_de2bi(D, N)
% Zamienia wektor liczb całkowitych D na binarne reprezentacje o N bitach
% (left-MSB: najstarszy bit pierwszy w wierszu)

D = D(:); % kolumna
B = zeros(length(D), N);
for i = 1:N
    B(:,i) = bitget(D, N - i + 1);
end
end

function D = my_bi2de(B, varargin)
% Zamienia macierz bitów B (każdy wiersz to jedna liczba binarna) na liczby dziesiętne
% Obsługuje opcjonalny argument 'left-msb' (domyślnie)

if nargin < 2
    order = 'right-msb';
else
    order = varargin{1};
end

[m, n] = size(B);
D = zeros(m,1);

if strcmpi(order, 'left-msb')
    for i = 1:m
        val = 0;
        for j = 1:n
            val = val + B(i,j) * 2^(n-j);
        end
        D(i) = val;
    end
else % right-msb
    for i = 1:m
        val = 0;
        for j = 1:n
            val = val + B(i,j) * 2^(j-1);
        end
        D(i) = val;
    end
end
end


