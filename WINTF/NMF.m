function [U, V]=NMF(R, param)
rng(7706,'twister');
[U, V] = updateUV(R, param);
end

function [U, V] = updateUV(R, para)
[m, n] = size(R);
rank = para(1);
reg=0.1;
maxIte = para(2);
U0 = rand(m, rank);
V0 = rand(n, rank);
ite=1;
while ite <maxIte
    U0 = updateU(R, U0, V0, reg);
    V0 = updateU(R', V0, U0, reg);
    ite = ite + 1;
end
U = U0;
V = V0;
end

function [U1] = updateU(R, U0, V, lambda)
U1 = U0.* ((R*V)./(U0*(V'*V) + (lambda.*U0) )) ;
U1(isnan(U1)) = 0;
end