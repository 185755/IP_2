clear; close all;
[input, fs] = audioread('voice.wav');
input = input(:,1);

N = 256;
r = 10;

m = 2; %quantization level
mLvl = m ^ 2; %posible levels 
window = 0.5 * (1 - cos((2*pi / (N+1)*(1:N))));
splited = zeros(floor(length(input)/246), 256);
step = N - r;

splited(1, :) = input(1:256)';
for i=2:(size(splited, 1))
    splited(i, :) = input((i - 1) * step:i * step + r - 1)';
end

flattened = zeros(size(splited));
for i=1:length(splited)
    flattened(i, :) = splited(i, :) .* window;
end

extended = zeros(size(flattened, 1), size(flattened, 2) + 20);
extended(:, r + 1 : N + r) = flattened;
for i= 1:size(extended, 1)
    R = xcorr(extended(i, :), r, 'biased');
    R = R(r+1:end);
    [a(i, :), sigma(i, :), k(i, :)] = L_D(R, r);
end

eMax = zeros(length(extended), 1);
for i = 1:length(eMax)
    e = zeros(N, 1);
    for j = 1 : N
        e(j) = extended(i, j+r) + a(i,:) * extended(i, j+r-1 : -1 : j)';
    end
    eMax(i) = max(abs(e));
end
xmin = min(eMax)
xmax = max(eMax)
delta = (xmax - xmin) / (mLvl - 1)

indices = round((eMax -xmin)/delta);
indices = max(0, min(indices, mLvl-1));

plot(indices * delta)
hold on;
plot(eMax)
hold off;

function [a, sigma, k] = L_D(pi, r)
    a = zeros(r, 1);
    sigma = zeros(r+1,1);
    k = zeros(r, 1);

    sigma(1) = pi(1);
    if sigma(1) == 0
        return;
    end

    k(1) = -pi(2)/pi(1);
    a(1) = k(1);
    sigma(2) = sigma(1)*(1- k(1)^2);

    for i = 2:r
        sum_k = 0;
        for j = 1:i-1
            sum_k = sum_k + a(j)*pi(i-j+1);
        end
        k(i) = -(pi(i+1)+sum_k)/sigma(i);
        a_old = a(1:i-1);
        for j = 1:i-1
            a(j) = a_old(j) + k(i) * conj(a_old(i-j));
        end
        a(i) = k(i);
        sigma(i+1) = sigma(i) * (1- k (i)^2);
    end
end