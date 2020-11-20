function [F,S,G]=nmtf(R,para)
rng(7706,'twister');
[F, S, G] = updateFSG(R, para);
end

function [F,S,G]=updateFSG(R, para)
[m,n]=size(R);
r=para(1); %first low-rank; F=(m,r)
s=para(2); %second low-rank; G=(n,r)
F0 = rand(m, r);
G0 = rand(n, s);
S0 = rand(r, s);
maxIte=para(3);
ite=0;
while ite <maxIte 
    F0 = updateF(R, F0, S0, G0);
    G0 = updateF(R', G0, S0', F0);
    S0 = updateS(R, F0, S0, G0);
    ite = ite + 1;
end
F = F0;
S = S0;
G = G0;
end

function [F1]=updateF(R,F0,S,G)
F1 = F0 .* sqrt((R*G*S')./(F0*S*G'*G*S'));
F1(isnan(F1)) = 0;
end

function [S1]=updateS(R,F,S0,G)
S1 = S0 .* sqrt((F'*R*G)./(F'*F*S0*G'*G));
S1(isnan(S1)) = 0;
end