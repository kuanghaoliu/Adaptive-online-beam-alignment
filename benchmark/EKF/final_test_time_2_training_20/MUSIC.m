function [alpha] = MUSIC(channel)
N = 64;
M = 1;
dd = 1 / 2;
d = 0 : dd : (N - 1) * dd;
derad = pi / 128;

X1 = channel;
Rxx = X1'*X1/N;
[EV,D] = eig(Rxx);    % eigenvalue and eigenvector
EVA = diag(D)';
[EVA,I] = sort(EVA);
EV = fliplr(EV(:,I));

for iang = 1 : 1 : 129 % loop for finding the maximum location
    angle(iang) = iang - 65;
    phim = derad * angle(iang);
    a = exp(1i * 2 * pi * d * sin(phim)).';
    L = M;
    En = EV(: , L + 1 : N);
    SP(iang) = 1 / (a' * En * En' * a);
end

[~, location] = max(SP); % find the maximum location
alpha = (location - 65) / 128 * pi;
    
end

