tic

%--------------------------------------
% Tear-down semua display dan variable
%--------------------------------------
clc; clear;

%--------------
% Load file GR 
%--------------
CM1_01_GR = csvread('03_SeleksiFitur\CM1_GR\CM1_GR.csv');

%-------------
% K-Fold = 10
%-------------
k = 10;
vektorCM1 = CM1_01_GR(:,1);
cvFolds = crossvalind('Kfold', vektorCM1, k);
clear vektorCM1;      
    
%------------------------------------------
% Pembagian data TESTING dan data TRAINING
%------------------------------------------
for iFold = 1 : k
%---
    %-------------------------------------
    % Penetapan data TRAINING dan TESTING
    %-------------------------------------
    testIdx = (cvFolds == iFold);                
    CM1_00_TrainIdx(:,iFold) = ~testIdx;

    %-----------------------------------------------------------
    % Pembagian data TRANING dan TESTING berdasarkan "trainIdx"
    %-----------------------------------------------------------
    iTraining = 1; 
    iTesting = 1; 
    jmlData = length(CM1_01_GR);
    for iBarisData = 1 : length(CM1_01_GR)
        %----------------------------------------------------------------
        % Mengambil urutan CM1_01_GR berdasarkan trainIdx = 1 [TRAINING]
        %----------------------------------------------------------------
        if CM1_00_TrainIdx(iBarisData,iFold) == 1                                          
            CM1_02_Train{iFold,1}(iTraining,1:22) = CM1_01_GR(iTraining,:); 
            CM1_02_Train{iFold,1}(iTraining,23) = iTraining; % tambah kolom keterangan urutan data
            iTraining = iTraining + 1;
        %-------------------------------------------------------------
        % Mengambil urutan (trainIdx ~= 1) dengan CM1_01_GR [TESTING]
        %-------------------------------------------------------------
        else                                         
            CM1_03_Test{iFold,1}(iTesting,:) = CM1_01_GR(iTesting,:);            
            iTesting = iTesting + 1;
        end
    end 
%---    
end
clear cvFolds iBarisData iFold iTesting iTraining testIdx k jmlData;




%----------------------------------------------------------------------------------------------------------
% Penentuan titik C1 yang mewakili kelas FALSE dan C2 yang mewakili kelas TRUE, berdasarkan "CM1_02_Train"
%----------------------------------------------------------------------------------------------------------
for iFold = 1 : 10  
%---
    %----------------------------------------------------
    % Pisahkan kelas TRUE dan FALSE pada "CM1_02_Train"
    %----------------------------------------------------
    fgFalse = 0;
    fgTrue = 0;
    for iJumlahTrain = 1 : length(CM1_02_Train{iFold,1})
        %----------------------------------------
        % Kalau kelas 0 maka jadi training FALSE
        %----------------------------------------
        if CM1_02_Train{iFold,1}(iJumlahTrain,22) == 0              
            fgFalse = fgFalse + 1;
            CM1_04_Train_False{iFold,1}(fgFalse,:) = CM1_02_Train{iFold,1}(iJumlahTrain,:); 
        %----------------------------------------
        % Kalau kelas 1 maka jadi training TRUE
        %----------------------------------------
        else                                 
            fgTrue = fgTrue + 1;
            CM1_05_Train_True{iFold,1}(fgTrue,:) = CM1_02_Train{iFold,1}(iJumlahTrain,:); 
        end
    end
    
    %---------------------------------------------------
    % Tentukan C1 dari kumpulan kelas FALSE secara acak
    %---------------------------------------------------   
    kFalse = randperm(length(CM1_04_Train_False{iFold,1})); % acak urutan data "trainingFalse"
    CM1_06_C1{iFold,1} = CM1_04_Train_False{iFold,1}(kFalse(1,1),:); % urutan pertama hasil acak, diambil sebagai C1
    clear kFalse;       

    %--------------------------------------------------
    % Tentukan C2 dari kumpulan kelas TRUE secara acak
    %--------------------------------------------------
    kTrue = randperm(length(CM1_05_Train_True{iFold,1})); % acak urutan data "trainingTrue"
    CM1_07_C2{iFold,1} = CM1_05_Train_True{iFold,1}(kTrue(1,1),:); % urutan pertama hasil acak, diambil sebagai C2
    clear kTrue;                    
%---
end
clear iFold fgFalse fgTrue iJumlahTrain;

% ==========================================================================================================================
%                                                     ==  WHILE  ===
% ==========================================================================================================================

%--------------------------------------------------------
% Hitung distance masing-masing fitur terhadap C1 dan C2
%--------------------------------------------------------
for iFold = 1 : 10
    for iKolomCluster = 1 : 21      
        for iBarisCluster = 1 : length(CM1_02_Train{iFold,1})  

            %------------------------------------
            % Hitung jarak data ke titik cluster
            %------------------------------------
            data = CM1_02_Train{iFold,1}(iBarisCluster,iKolomCluster);

            %------------------------
            % Jarak tiap fitur ke C1
            %------------------------
            C1 = CM1_06_C1{iFold,1}(1,iKolomCluster);                                
            jarakHamming = hammingDistance_fix(data,C1);
            CM1_08_HamDist_C1{iFold,1}(iBarisCluster,iKolomCluster) = jarakHamming;

            %------------------------
            % Jarak tiap fitur ke C2
            %------------------------
            C2 = CM1_07_C2{iFold,1}(1,iKolomCluster);                                
            jarakHamming = hammingDistance_fix(data,C2);
            CM1_09_HamDist_C2{iFold,1}(iBarisCluster,iKolomCluster) = jarakHamming;                                   
        
        end                        
    end  
%---
end
clear C1 C2 data iBarisCluster iKolomCluster jarakHamming iFold;

%-------------------------------------------------------------------------------------------------
% Ngambil semua distance di "CM1_08_HamDist_C1" dan "CM1_09_HamDist_C2" berdasarkan seleksi fitur
%-------------------------------------------------------------------------------------------------
for iSeleksiFitur = 1 : 21    
%---    
    for iFold = 1 : 10
        for iBarisSF = 1 : length(CM1_02_Train{iFold,1})  
            CM1_10_SF_C1{1,iSeleksiFitur}{iFold,1}(iBarisSF,:) = CM1_08_HamDist_C1{iFold,1}(iBarisSF,1:iSeleksiFitur);
            CM1_11_SF_C2{1,iSeleksiFitur}{iFold,1}(iBarisSF,:) = CM1_09_HamDist_C2{iFold,1}(iBarisSF,1:iSeleksiFitur);
        end                
    end
%---
end
clear iBarisSF iSeleksiFitur iFold;

%-----------------------------------------------------------
% Menghitung rata-rata setiap baris data pada seleksi fitur
%-----------------------------------------------------------
for iFitur = 1 : 21
%---
    for iFold = 1 : 10  
        for iBarisAvg = 1 : length(CM1_02_Train{iFold,1})              
            totalC1 = 0;
            totalC2 = 0;            
            for iSeleksiFitur = 1 : iFitur                                                           

                %---------------------------------------
                % Jumlah distance C1 dari setiap fitur
                %---------------------------------------
                nilaiC1 = CM1_10_SF_C1{1,iFitur}{iFold,1}(iBarisAvg,iSeleksiFitur);
                totalC1 = totalC1 + nilaiC1;                

                %---------------------------------------
                % Jumlah distance C2 dari setiap fitur
                %---------------------------------------
                nilaiC2 = CM1_11_SF_C2{1,iFitur}{iFold,1}(iBarisAvg,iSeleksiFitur);
                totalC2 = totalC2 + nilaiC2;                                                            
            
            end
            
            %-------------------
            % Total distance C1
            %-------------------
            averageC1 = totalC1 / iFitur;
            CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisAvg,1) = averageC1;   
            
            %-------------------
            % Total distance C2
            %-------------------
            averageC2 = totalC2 / iFitur;
            CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisAvg,2) = averageC2;
            
            %------------------------------
            % Penentuan anggota C1 atau C2
            %------------------------------
            if averageC1 > averageC2                
                CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisAvg,3) = 22222;
            else CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisAvg,3) = 11111;
            end  
            
        end
    end
%---   
end
clear averageC1 averageC2 iBarisAvg iFitur iFold nilaiC1 nilaiC2 totalC1 totalC2 iSeleksiFitur;

%----------------------------------------------------------------
% Pembagian anggota C1 dan C2 berdasarkan "CM1_12_TotDist_C1_C2"
%----------------------------------------------------------------
for iFitur = 1 : 21    
    for iFold = 1 : 10            
        fgC1 = 0;
        fgC2 = 0;
        for iBarisBagi = 1 : length(CM1_02_Train{iFold,1})  
            if CM1_12_TotDist_C1_C2{1,iFitur}{iFold,1}(iBarisBagi,3) == 11111     
                fgC1 = fgC1 + 1;
                CM1_13_Anggota_C1{1,iFitur}{iFold,1}(fgC1,:) = CM1_02_Train{iFold,1}(iBarisBagi,1:iFitur);                
            else
                fgC2 = fgC2 + 1;
                CM1_14_Anggota_C2{1,iFitur}{iFold,1}(fgC2,:) = CM1_02_Train{iFold,1}(iBarisBagi,1:iFitur);
            end                        
        end
        
        %-----------------------------------------------------------------------------------------------
        % Prevent Fold < 10 untuk anggota C2, jadi metrik kosong di akhir dianggap tidak ada sama matLab    
        %-----------------------------------------------------------------------------------------------
        if length(CM1_13_Anggota_C1{1,iFitur}{iFold,1}) == length(CM1_02_Train{iFold,1})
            CM1_14_Anggota_C2{1,iFitur}{iFold,1} = [];
        end
        
    end
end
clear iBarisBagi iFitur fgC1 fgC2 iFold;

%-----------------------------------
% MEAN fitur dari anggota C1 dan C2
%-----------------------------------
for iFitur = 1 : 21    
    for iFold = 1 : 10    
        %-------------------------------------------
        % Hitung MEAN "CM1_13_Anggota_C1" per fitur 
        %-------------------------------------------
        CM1_15_Mean_C1{1,iFitur}{iFold,1}(1,:) = mean(CM1_13_Anggota_C1{1,iFitur}{iFold,1});              
        
        %-----------------------------------------------------------------------------------------------
        % Prevent Fold < 10 untuk anggota C2, jadi metrik kosong di akhir dianggap tidak ada sama matLab    
        %-----------------------------------------------------------------------------------------------
        if length(CM1_13_Anggota_C1{1,iFitur}{iFold,1}) == length(CM1_02_Train{iFold,1})
            CM1_16_Mean_C2{1,iFitur}{iFold,1} = [];
        end
        
        %-------------------------------------------
        % Hitung MEAN "CM1_14_Anggota_C2" per fitur 
        %------------------------------------------- 
        if length(CM1_14_Anggota_C2{1,iFitur}) ~= 0            
            if length(CM1_14_Anggota_C2{1,iFitur}{iFold,1}) ~= 0                  
                %-------------------------------------------------------------
                % Kondisi kalau baris datanya cuma 1, jadi ga usah hitung mean
                %-------------------------------------------------------------
                if size(CM1_14_Anggota_C2{1,iFitur}{iFold,1},1) == 1
                    CM1_16_Mean_C2{1,iFitur}{iFold,1}(1,:) = CM1_14_Anggota_C2{1,iFitur}{iFold,1};
                else
                    CM1_16_Mean_C2{1,iFitur}{iFold,1}(1,:) = mean(CM1_14_Anggota_C2{1,iFitur}{iFold,1});       
                end                  
            end            
        end  
        
    end
end
clear iFitur iFold;

%-----------------------
% Pembulatan nilai MEAN
%-----------------------
for iFitur = 1 : 21
    for iFold = 1 : 10
        for iSeleksiFitur = 1 : iFitur            
            
            %--------------------------
            % Pembulatan nilai MEAN C1
            %--------------------------
            nilaiMeanC1 = CM1_15_Mean_C1{1,iFitur}{iFold,1}(1,iSeleksiFitur);
            pembulatanC1 = pembulatanMEAN_fix(nilaiMeanC1);
            CM1_17_MeanBulat_C1{1,iFitur}{iFold,1}(1,iSeleksiFitur) = pembulatanC1;
            
            %--------------------------
            % Pembulatan nilai MEAN C2
            %--------------------------
            if length(CM1_16_Mean_C2{1,iFitur}) ~= 0
                if length(CM1_16_Mean_C2{1,iFitur}{iFold,1}) ~= 0
                    nilaiMeanC2 = CM1_16_Mean_C2{1,iFitur}{iFold,1}(1,iSeleksiFitur);
                    pembulatanC2 = pembulatanMEAN_fix(nilaiMeanC2);
                    CM1_18_MeanBulat_C2{1,iFitur}{iFold,1}(1,iSeleksiFitur) = pembulatanC2;
                end
            end 
            
            %-----------------------------------------------------------------------------------------------
            % Prevent Fold < 10 untuk anggota C2, jadi metrik kosong di akhir dianggap tidak ada sama matLab    
            %-----------------------------------------------------------------------------------------------
            if length(CM1_13_Anggota_C1{1,iFitur}{iFold,1}) == length(CM1_02_Train{iFold,1})
                CM1_18_MeanBulat_C2{1,iFitur}{iFold,1} = [];
            end
            
        end        
    end
end
clear iFitur iFold iSeleksiFitur nilaiMeanC1 nilaiMeanC2 pembulatanC1 pembulatanC2 pembulatanMEAN_fix;

%-----------------------------------------------------------
% Pembagian titik C1 dan C2 (awal) berdasarkan seleksi fitur
%-----------------------------------------------------------
for iFitur = 1 : 21
    for iFold = 1 : 10                
        CM1_19_C1_SeleksiFitur{1,iFitur}{iFold,1}(1,1:iFitur) = CM1_06_C1{iFold,1}(1:1:iFitur);                        
        CM1_20_C2_SeleksiFitur{1,iFitur}{iFold,1}(1,1:iFitur) = CM1_07_C2{iFold,1}(1:1:iFitur);        
    end
end
clear iFitur iFold;

% --------------------------------------------------------------
% Pembagian data TRAINING dan TESTING berdasarkan seleksi fitur
% --------------------------------------------------------------
for iFitur = 1 : 21
    for iFold = 1 : 10
        CM1_21_Train_SeleksiFitur{1,iFitur}{iFold,1}(1,1:iFitur) = CM1_02_Train{iFold,1}(1:1:iFitur);        
        CM1_22_Test_SeleksiFitur{1,iFitur}{iFold,1}(1,1:iFitur) = CM1_03_Test{iFold,1}(1:1:iFitur);        
    end
end
clear iFitur iFold;

% ==========================================================================================================================
%                                                     ==  WHILE  ===
% ==========================================================================================================================

%---------------------------------------------------------------------------------
% 1. Cek apakah titik C1 yang lama dengan yang baru sudah sama? If ya = konvergen
% 2. If tidak = Hitung lagi, hingga anggota C1 dan C2 muncul yang baru
% 3. Update nilai C1 dan C2 berdasarkan MEAN dari proses kedua (2.)
%---------------------------------------------------------------------------------
% for iFitur = 1 : 21
%     for iFold = 1 : 10        
%                                     
%         %--------------------------------------
%         % 1. while titik C1 LAMA != titik C1 BARU
%         %--------------------------------------
%         while(CM1_17_MeanBulat_C1{1,iFitur}{iFold,1} ~= CM1_19_C1_SeleksiFitur{1,iFitur}{iFold,1})
%                 
%             CM1_21_C1_temp_A{1,iFitur}{iFold,1} = CM1_17_MeanBulat_C1{1,iFitur}{iFold,1};
%             
%             %--------------------------------------------
%             % 2. Hitung lagi anggota C1 dan C2 yang baru
%             %--------------------------------------------
%             for iBaris = 1 : length(CM1_02_Train)
%                 
%                 %----------------------------------------------
%                 % Hitung jarak data setiap fitur ke titik C1 dan C2
%                 %----------------------------------------------
%                 data = CM1_02_Train{iFold,1}(iBaris,iFitur);
% 
%                 %------------------------
%                 % Jarak tiap fitur ke C1
%                 %------------------------
%                 C1 = CM1_21_C1_temp_A{1,iFitur}{iFold,1};
%                 C1 = CM1_06_C1{iFold,1}(1,iKolomCluster);                                
%                 jarakHamming = hammingDistance_fix(data,C1);
%                 CM1_08_HamDist_C1{iFold,1}(iBarisCluster,iKolomCluster) = jarakHamming;
% 
% %             %------------------------
% %             % Jarak tiap fitur ke C2
% %             %------------------------
% %             C2 = CM1_07_C2{iFold,1}(1,iKolomCluster);                                
% %             jarakHamming = hammingDistance_fix(data,C2);
% %             CM1_09_HamDist_C2{iFold,1}(iBarisCluster,iKolomCluster) = jarakHamming;
%                 
%                 
%             end
%             
%             
% 
% 
%         end        
%     end
% end    

toc