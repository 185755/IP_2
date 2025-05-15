clear all;
clc;

[x1, fs] = audioread("pan_tadeusz1.wav");
x1 = x1(:,1);
%x1 = int16(x1*32767);

podzielone = podzial(x1);
r = 10;
params = zeros(r,1);

splaszczone = splaszcz(podzielone);
wyzerowane_konce = wyzeruj_konce(splaszczone);
wymiary = size(wyzerowane_konce);
wektory_korelacji = zeros(wymiary(1),r+1);
wektor_p0 = zeros(wymiary(1),1);
[pi, Ri] = policz_pi_Ri(wyzerowane_konce(1,:),r);

for i = 1:wymiary(1)
    wektor_p0(i,1) = policz_p0(wyzerowane_konce(i,:),r);
    [wektory_korelacji(i,:),~] = policz_pi_Ri(wyzerowane_konce(i,:),r);
end

[k, a, e] = L_D(wektory_korelacji(1,:));

% [amat,emat,kmat] = levinson(wektory_korelacji(1,:),10);





%player1 = audioplayer(x1,fs) ;
%play(player1);

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

function [p0] = policz_p0(y, r)
    suma = zeros(1, r-1);
    p0 = 0;
    N = size(y);
    for t = 1 : N(2)
            p0 = p0 + y(t)*y(t);
    end
end
%p0 = pi(1) p1 = pi(2)
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