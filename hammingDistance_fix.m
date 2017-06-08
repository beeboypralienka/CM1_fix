function jarakHamming = hammingDistance_fix(data,C)

% ----------------------------
% Bikin kemungkinan kombinasi
% ----------------------------
if data == C    
    jarakHamming = 0;
elseif (data == 0 && C == 3) || (data == 3 && C == 0) ||(data == 1 && C == 2) || (data == 2 && C == 1)    
    jarakHamming = 2/2;
else jarakHamming = 1/2;
end