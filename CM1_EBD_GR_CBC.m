tic

% -------------------------------------
% Tear-down semua display dan variable
% -------------------------------------
clc; clear;

% ------------------------
% Load file GR Fold 1 - 10
% ------------------------
CM1_01_GR = csvread('03_SeleksiFitur\CM1_GR\CM1_GR.csv');

% --------------------------------
% Pembagian dataset secara manual:
% TRAINING (66%)
% TESTING (33%)
% --------------------------------
jmlDataset = length(CM1_01_GR);
perTraining = jmlDataset * (66/100);
jmlTraining = floor(perTraining); % Pembulatan ke bawah
jmlTesing = jmlDataset - jmlTraining;

% -------------------------------------------------------
% jmlTesting dibagi dua untuk ngambil dari atas dan bawah
% -------------------------------------------------------
ambil = jmlTesing/2;
ambil = floor(ambil);

% ---------------------------------------------------------------------------------
% Testing ngambilnya dari atas dan bawah dataset, sisannya adalah training (tengah)
% ---------------------------------------------------------------------------------
iTesting = 0;
iTrain = 0;
for iPembagian = 1 : jmlDataset
    if (iPembagian > ambil) && (iPembagian <= jmlDataset-ambil)
        iTrain = iTrain + 1;
        CM1_02_Train(iTrain,:) = CM1_01_GR(iPembagian,:);  
    else
        iTesting = iTesting + 1;
        CM1_03_Test(iTesting,:) = CM1_01_GR(iPembagian,:);  
    end
end
clear iBagi iBagi2 iBagiTrain iTesting jmlTraining jmlTesing ambil iPembagian iTrain jmlDataset perTraining;

% % ------------
% % K-Fold = 10
% % ------------
% k = 10;
% vektorCM1 = CM1_01_GR(:,1);
% cvFolds = crossvalind('Kfold', vektorCM1, k);
% clear vektorCM1;      
%     
% % -------------
% % Iterasi FOLD
% % -------------
% for iFold = 1 : k
% %---
% 
%     % -----------------------------------
%     % Penetapan data TRAINING dan TESTING
%     % -----------------------------------
%     testIdx = (cvFolds == iFold);                
%     CM1_00_TrainIdx(:,iFold) = ~testIdx;
% 
%     % ---------------------------------------------------------
%     % Pembagian data TRANING dan TESTING berdasarkan "trainIdx"
%     % ---------------------------------------------------------
%     iTraining = 1; 
%     iTesting = 1; 
%     for iBarisData = 1 : length(CM1_01_GR)                            
%         % --------------------------------------------------------------
%         % Mengambil urutan CM1_01_GR berdasarkan trainIdx = 1 [TRAINING]
%         % --------------------------------------------------------------
%         if CM1_00_TrainIdx(iBarisData,iFold) == 1                                          
%             CM1_02_Train{iFold,1}(iTraining,1:22) = CM1_01_GR(iTraining,:); 
%             CM1_02_Train{iFold,1}(iTraining,23) = iTraining; % tambah ket. urutan
%             iTraining = iTraining + 1;
%         % -----------------------------------------------------------
%         % Mengambil urutan (trainIdx ~= 1) dengan CM1_01_GR [TESTING]
%         % -----------------------------------------------------------
%         else                                         
%             CM1_03_Test{iFold,1}(iTesting,:) = CM1_01_GR(iTesting,:);            
%             iTesting = iTesting + 1;
%         end
%     end       
% 
%     % --------------------------------------------------
%     % Pisahkan kelas TRUE dan FALSE pada "CM1_02_Train"
%     % --------------------------------------------------
%     fgFalse = 0;
%     fgTrue = 0;
%     for iJumlahData = 1 : length(CM1_02_Train{iFold,1})                          
%         % ---------------------------------------------
%         % Kalau kelas 0 maka jadi training khusus FALSE
%         % ---------------------------------------------
%         if CM1_02_Train{iFold,1}(iJumlahData,22) == 0            
%             fgFalse = fgFalse + 1;
%             CM1_04_Train_False{iFold,1}(fgFalse,:) = CM1_02_Train{iFold,1}(iJumlahData,:);                    
%         % --------------------------------------------
%         % Kalau kelas T maka jadi training khusus TRUE
%         % --------------------------------------------    
%         else                         
%             fgTrue = fgTrue + 1;
%             CM1_05_Train_True{iFold,1}(fgTrue,:) = CM1_02_Train{iFold,1}(iJumlahData,:);            
%         end                         
%     end        
%         
%     % ------------------------------------------------
%     % Tentukan C1 dari kumpulan kelas FALSE secara acak
%     % ------------------------------------------------    
%     kFalse = randperm(length(CM1_04_Train_False{iFold,1})); % acak urutan data "trainingFalse"
%     CM1_06_C1 = CM1_04_Train_False{iFold,1}(kFalse(1,1),:); % urutan pertama hasil acak, diambil sebagai C1
% 
%     % -------------------------------------------------
%     % Tentukan C2 dari kumpulan kelas TRUE secara acak
%     % ------------------------------------------------
%     kTrue = randperm(length(CM1_05_Train_True{iFold,1})); % acak urutan data "trainingTrue"
%     CM1_07_C2 = CM1_05_Train_True{iFold,1}(kTrue(1,1),:); % urutan pertama hasil acak, diambil sebagai C2
%         
%     % ------------------------------------------------------
%     % Hitung distance masing-masing fitur terhadap C1 dan C2
%     % ------------------------------------------------------
%     for iKolomCluster = 1 : 21        
%         for iBarisCluster = 1 : length(CM1_02_Train{iFold,1})            
%             % ----------------------------------
%             % Hitung jarak data ke titik cluster
%             % ----------------------------------
%             data = CM1_02_Train{iFold,1}(iBarisCluster,iKolomCluster);
%                 
%             % ----------------------
%             % Jarak tiap fitur ke C1
%             % ----------------------
%             C1 = CM1_06_C1(1,iKolomCluster);                                
%             jarakHamming = hammingDistance_fix(data,C1);
%             CM1_08_HamDist_C1{iFold,1}(iBarisCluster,iKolomCluster) = jarakHamming;
%                 
%             % ----------------------
%             % Jarak tiap fitur ke C2
%             % ----------------------
%             C2 = CM1_07_C2(1,iKolomCluster);                                
%             jarakHamming = hammingDistance_fix(data,C2);
%             CM1_09_HamDist_C2{iFold,1}(iBarisCluster,iKolomCluster) = jarakHamming;                                   
%         end                        
%     end                              
% %--- 
% end
% clear C1 C2 cvFolds data fgFalse fgTrue iBarisCluster iBarisData iFold iJumlahData iKolomCluster;
% clear iTesting iTraining jarakHamming k kFalse kTrue testIdx;
%     
% % -----------------------------------------------------------------------------------------------
% % Ngambil semua distance di "CM1_08_HamDist_C1" dan "CM1_09_HamDist_C2" berdasarkan seleksi fitur
% % -----------------------------------------------------------------------------------------------
% for iSeleksiFitur = 1 : 21    
%     for iFoldSF = 1 : 10        
%         for iBarisSF = 1 : length(CM1_02_Train{iFoldSF,1})
%             CM1_10_SF_C1{1,iSeleksiFitur}{iFoldSF,1}(iBarisSF,:) = CM1_08_HamDist_C1{iFoldSF,1}(iBarisSF,1:iSeleksiFitur);
%             CM1_11_SF_C2{1,iSeleksiFitur}{iFoldSF,1}(iBarisSF,:) = CM1_09_HamDist_C2{iFoldSF,1}(iBarisSF,1:iSeleksiFitur);
%         end            
%     end
% end
% clear iBarisSF iFoldSF iSeleksiFitur;
% 
% % -----------------------------------------------------------------------
% % Menghitung rata-rata satu baris dari setiap baris data di seleksi fitur
% % -----------------------------------------------------------------------
% for iFitur = 1 : 21
%     for iFold = 1 : 10
%         for iBarisAvg = 1 : length(CM1_02_Train{iFold,1})  
%             totalC1 = 0;
%             totalC2 = 0;
%                         
%             % ---------------------------------------
%             % Jumlah distance ke C1 dari setiap fitur
%             % ---------------------------------------
%             nilaiC1 = CM1_10_SF_C1{1,iFitur}{iFold,1}(iBarisAvg,iFitur);
%             totalC1 = totalC1 + nilaiC1;
%                 
%             % ---------------------------------------
%             % Jumlah distance ke C2 dari setiap fitur
%             % ---------------------------------------
%             nilaiC2 = CM1_11_SF_C2{1,iFitur}{iFold,1}(iBarisAvg,iFitur);
%             totalC2 = totalC2 + nilaiC2;            
%                         
%             % -----------------
%             % Total distance C1
%             % -----------------
%             averageC1 = totalC1 / iFitur;
%             CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisAvg,1) = averageC1;
%             
%             % -----------------
%             % Total distance C2
%             % -----------------
%             averageC2 = totalC2 / iFitur;
%             CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisAvg,2) = averageC2;
%             
%             % ----------------------------
%             % Penentuan anggota C1 atau C2
%             % ----------------------------
%             if averageC1 > averageC2
%                 CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisAvg,3) = 22222;
%             else CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisAvg,3) = 11111;
%             end            
%         end
%     end
% end
% clear averageC1 averageC2 iBarisAvg iFitur iFold nilaiC1 nilaiC2 totalC1 totalC2;
% 
% % --------------------------------------------
% % Membagi data ke kelas 22222 atau kelas 11111
% % --------------------------------------------
% for iFitur = 1 : 21
%     for iFold = 1 : 10
%         for iBarisBagi = 1 : length(CM1_02_Train{iFold,1})
%             if CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisBagi,3) == 11111
%                 CM1_13_Anggota_C1{1,iFitur}{iFold,1}(iBarisBagi,:) = CM1_02_Train{iFold,1}(iBarisBagi,1:iFitur);
%             else
%                 CM1_13_Anggota_C2{1,iFitur}{iFold,1}(iBarisBagi,:) = CM1_02_Train{iFold,1}(iBarisBagi,1:iFitur);
%             end
%         end
%     end
% end
% clear iBarisBagi iFitur iFold;

toc