function entropyParent = entropyParentEBD(jmlTrue,jmlFalse,jmlData)

% Rumus menghitung entropy parent di tahap EBD
piTrue = jmlTrue / jmlData;
piFalse = jmlFalse / jmlData;
Log2piTrue = log2(piTrue);
Log2piFalse = log2(piFalse);
kaliLogTrue = Log2piTrue * piTrue;
kaliLogFalse = Log2piFalse * piFalse;
entropyParent = abs (kaliLogTrue + kaliLogFalse);        