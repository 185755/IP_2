plot(reconstructed);
hold on
plot(audio);
for i = 1 : 934
    xline(i*256);
end
