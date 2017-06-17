function pembulatan = pembulatanMEAN_fix(data)

% ---------------
% RULE pembulatan
% ---------------
% N <= 0.5 = 0
% N <= 1.5 = 1
% N <= 2.5 = 2
% N  > 2.5 = 3 
% ---------------
if data <= 0.5    
    pembulatan = 0;
elseif data <= 1.5
    pembulatan = 1;
elseif data <= 2.5
    pembulatan = 2;
else pembulatan = 3;
end