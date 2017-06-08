tic

% -------------------------------------
% Tear-down semua display dan variable
% -------------------------------------
clc; clear;

% -----------------------------
% Load file CSV dataset mentah
% -----------------------------
CM1_01_Dataset = csvread('01_Data\CM1.csv');

% ----------------------------------------
% Load file CSV dataset (remove duplicate)
% ----------------------------------------
CM1_02_Unique = csvread('01_Data\CM1Unique.csv');

%==================================================================================================================================================================================
%                                              ********************** EBD FASE 1 **********************
%==================================================================================================================================================================================

% ----------------------------
% Hitung jumlah TRUE dan FALSE
% ----------------------------
jmlTrue = 0;
jmlFalse = 0;
for iJumlahTF = 1 : length(CM1_02_Unique)
    % -------------------
    % Hitung jumlah FALSE
    % -------------------
    if CM1_02_Unique(iJumlahTF,22) == 0
        jmlTrue = jmlTrue + 1;
    % -------------------
    % Hitung jumlah TRUE
    % -------------------
    else
        jmlFalse = jmlFalse + 1;
    end    
end

% ------------------------------------------------------------------------------
% Menghitung entropy parent di tahap EBD (menggunakan fungsi "entropyParentEBD")
% ------------------------------------------------------------------------------
jmlData = length(CM1_02_Unique);
entropyParent = entropyParentEBD_fix(jmlTrue,jmlFalse,jmlData);

% ------------------------------
% Rangkuman "CM1_03_Keterangan"
% ------------------------------
% [1] Jumlah Data
% [2] TRUE
% [3] FALSE
% [4] Entropy Parent
% --------------------------------------------------------
CM1_03_Keterangan = [jmlData jmlTrue jmlFalse entropyParent];

% ----------------------------------------------
% Data dipecah per fitur, kemudian diurutkan ASC
% ----------------------------------------------
for iFiturPecah = 1 : 21
    for iDataPecah = 1 : jmlData    
        CM1_04_DataFitur{1,iFiturPecah}(iDataPecah,:) = [CM1_02_Unique(iDataPecah,iFiturPecah) CM1_02_Unique(iDataPecah,22)];
        CM1_05_DataFiturASC{1,iFiturPecah} = sortrows(CM1_04_DataFitur{1,iFiturPecah});
    end    
end

% -------------------------------
% Data "CM1_05_DataFiturASC" di-split
% -------------------------------
for iFiturSplit = 1 : 21    
    for iDataSplit = 1 : jmlData - 1
        dataPertama = CM1_05_DataFiturASC{1,iFiturSplit}(iDataSplit,1); % Urutan iDataSplit
        dataKedua = CM1_05_DataFiturASC{1,iFiturSplit}(iDataSplit+1,1); % Urutan iDataSplit + 1
        CM1_06_DataFiturASC_Split{1,iFiturSplit}(iDataSplit,1) = (dataPertama+dataKedua)/2; % Ditambah dan dibagi dua, nilainya disimpan di kolom 1                   
    end
end

% --------------------------------------
% Kolom "CM1_06_DataFiturASC_Split" be like:
% --------------------------------------
% [1] Data Split
% [2] TRUE(<=)
% [3] FALSE(<=)
% [4] entropyChildren(<=)
% [5] TRUE(>)
% [6] FALSE(>)
% [7] entropyChildren(>)
% [8] INFO
% [9] GAIN
% --------------------------------------

% -------------------------------------------
% Proses update kolom "CM1_06_DataFiturASC_Split"
% -------------------------------------------
for iKolomUpdate = 1 : 21
%--
    for iBarisSplit = 1 : jmlData - 1        
        jmlTrueKurang = 0;
        jmlFalseKurang = 0;
        jmlTrueLebih = 0;
        jmlFalseLebih = 0;        
        for iBarisASC = 1 : jmlData            
            % -----------------------------------------------------------
            % Hitung jumlah TRUE dan FALSE dari kategoti ( <= ) dan ( > )
            % -----------------------------------------------------------
            dataSplit = CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,1);
            dataASC = CM1_05_DataFiturASC{1,iKolomUpdate}(iBarisASC,1);
            kelasASC = CM1_05_DataFiturASC{1,iKolomUpdate}(iBarisASC,2);
            % -------------
            % Kategori <=
            % -------------
            if dataASC <= dataSplit
                if kelasASC == 1                    
                    jmlTrueKurang = jmlTrueKurang + 1;
                else jmlFalseKurang = jmlFalseKurang + 1;
                end
            % ------------
            % Kategori >
            % ------------
            else
                if kelasASC == 1
                    jmlTrueLebih = jmlTrueLebih + 1;
                else jmlFalseLebih = jmlFalseLebih + 1;
                end
            end                                                            
        end        
        
        % -------------------------------------------------------
        % Update kolom "CM1_06_DataFiturASC_Split" [2], [3], [5], [6]
        % -------------------------------------------------------
        CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,2) = jmlTrueKurang;
        CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,3) = jmlFalseKurang;
        CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,5) = jmlTrueLebih;
        CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,6) = jmlFalseLebih;    
        
        % ------------------------------------------
        % Hitung ENTROPY CHILD dari parameter ( <= )
        % ------------------------------------------
        totalKurang = jmlTrueKurang + jmlFalseKurang; % Total jumlah TRUE dan jumlah FALSE dari parameter ( <= )
        % ----------------------------------------------------------------
        % Selama total jumlah TRUE dan FALSE di parameter ( <= ) bukan NOL
        % ----------------------------------------------------------------
        if totalKurang ~=0 
            piTrueKurang(iBarisSplit,1) = jmlTrueKurang / (jmlTrueKurang+jmlFalseKurang); % Hitung Pi TRUE ( <= )
            piFalseKurang(iBarisSplit,1) = jmlFalseKurang / (jmlTrueKurang+jmlFalseKurang); % Hitung Pi FALSE ( <= )
            % --------------------------------------------------------------------------------
            % Jika hasil Pi TRUE atau Pi FALSE itu NOL, dipastikan ENTROPY CHILD (<=) juga NOL            
            % --------------------------------------------------------------------------------
            if piTrueKurang(iBarisSplit,1) == 0 || piFalseKurang(iBarisSplit,1) == 0                 
                entropyChildKurang(iBarisSplit,1) = 0; % Entropy child ( <= ) dijadikan NOL
            % ---------------------------------------------------------------------
            % Jika Pi TRUE dan Pi FALSE bukan NOL, maka hitung ENTROPY CHILD ( <= )
            % ---------------------------------------------------------------------
            else entropyChildKurang = entropyChildrenEBD_fix(piTrueKurang, piFalseKurang,iBarisSplit);
            end                
        % ---------------------------------------------------------------------------------------------------------        
        % Jika total jumlah TRUE dan FALSE di parameter ( <= ) adalah NOL, dipastikan ENTROPY CHILD ( <= ) juga NOL
        % ---------------------------------------------------------------------------------------------------------
        else entropyChildKurang(iBarisSplit,1) = 0; % Entropy child ( <= ) dijadikan NOL
        end                           
                        
        % ------------------------------------------
        % Hitung ENTROPY CHILD dari parameter ( > )
        % ------------------------------------------
        totalLebih = jmlTrueLebih + jmlFalseLebih; % Total jumlah TRUE dan jumlah FALSE dari parameter ( > )
        % ----------------------------------------------------------------
        % Selama total jumlah TRUE dan FALSE di parameter ( > ) bukan NOL
        % ----------------------------------------------------------------
        if totalLebih ~= 0
            piTrueLebih(iBarisSplit,1) = jmlTrueLebih /  (jmlTrueLebih+jmlFalseLebih); % Hitung Pi TRUE ( > )
            piFalseLebih(iBarisSplit,1) = jmlFalseLebih / (jmlTrueLebih+jmlFalseLebih); % Hitung Pi FALSE ( > )                
            % --------------------------------------------------------------------------------
            % Jika hasil Pi TRUE atau Pi FALSE itu NOL, dipastikan entropyChild ( > ) juga NOL
            % --------------------------------------------------------------------------------
            if piTrueLebih(iBarisSplit,1) == 0 || piFalseLebih(iBarisSplit,1) == 0
                   entropyChildLebih(iBarisSplit,1) = 0; % Entropy child ( > ) dijadikan NOL            
            % ---------------------------------------------------------------------
            % Jika Pi TRUE dan Pi FALSE bukan NOL, maka hitung ENTROPY CHILD ( > )
            % ---------------------------------------------------------------------
            else entropyChildLebih = entropyChildrenEBD_fix(piTrueLebih, piFalseLebih,iBarisSplit);                   
            end
        % ----------------------------------------------------------------
        % Jika total jumlah TRUE dan FALSE adalah NOL pada parameter ( > )                
        % ----------------------------------------------------------------
        else entropyChildLebih(iBarisSplit,1) = 0; % Entropy child ( > ) dijadikan NOL
        end            
                                                            
        % ------------------------------------------------
        % Update kolom "CM1_06_DataFiturASC_Split" [4] dan [7]
        % ------------------------------------------------
        CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,4) = entropyChildKurang(iBarisSplit,1); % Ent. Child <=
        CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,7) = entropyChildLebih(iBarisSplit,1); % Ent. Child >                      
        
        % -----------------------------------------
        % Mencari nilai INFO dari setiap data split
        % -----------------------------------------
        dataChildKurang = (totalKurang/jmlData) * CM1_06_DataFiturASC_Split{1, iKolomUpdate}(iBarisSplit,4);
        dataChildLebih = (totalLebih/jmlData) * CM1_06_DataFiturASC_Split{1, iKolomUpdate}(iBarisSplit,7);
        INFOsplit(iBarisSplit,1) = (dataChildKurang + dataChildLebih);        
            
        % ------------------------------------
        % Mencari nilai GAIN dari setiap INFO
        % ------------------------------------
        GAINinfo(iBarisSplit,1) = CM1_03_Keterangan(1,4) - INFOsplit(iBarisSplit,1);                
        
        % ------------------------------------------------
        % Update kolom "CM1_06_DataFiturASC_Split" [8] dan [9]
        % ------------------------------------------------
        CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,8) = INFOsplit(iBarisSplit,1); % nilai INFO dari data SPLIT
        CM1_06_DataFiturASC_Split{1,iKolomUpdate}(iBarisSplit,9) = GAINinfo(iBarisSplit,1); % nilai INFO dari data SPLIT
    end
    
    % ---------------------------------
    % Distinct "CM1_06_DataFiturASC_Split"
    % ---------------------------------    
    CM1_07_DataFiturASC_Split_Distinct{1,iKolomUpdate} = unique(CM1_06_DataFiturASC_Split{1,iKolomUpdate},'rows');    
        
    % -------------------------------------------------------------------------------------------
    % Mencari nilai BEST SPLIT berdasarkan nilai GAIN tertinggi (max) di "Mtraining02UrutSplit_1"
    % -------------------------------------------------------------------------------------------
    [Nilai,BarisKe] = max(CM1_07_DataFiturASC_Split_Distinct{1,iKolomUpdate}(:,9)); % Ambil urutan ke berapa si split terbaik itu dan ambil nilai max gain-nya
    angkaBestSplit = CM1_07_DataFiturASC_Split_Distinct{1, iKolomUpdate}(BarisKe,1); % Angka split terbaik
    CM1_08_BEST_Split_1{1,iKolomUpdate} = [BarisKe angkaBestSplit Nilai]; % nilai max Gain dari data split ke berapa    
    
    % ----------------------------------------
    % Keterangan kolom "CM1_08_BEST_Split_1":
    % ----------------------------------------
    % [1] Baris ke berapa
    % [2] Angka BEST SPLIT
    % [3] Nilai Max GAIN
    % ----------------------------------------
%--
end

% --------------------------------------------------------------------------------
% Diskritisasi data "CM1_04_DataFitur" menjadi BINER berdasakan best split ( <= , > )
% --------------------------------------------------------------------------------
for iKolomDiskrit = 1 : 21 % Iterasi fitur exclude kelas    
    for iBarisDiskrit = 1 : jmlData
        dataAwal = CM1_04_DataFitur{1, iKolomDiskrit}(iBarisDiskrit,1);
        bestSplit = CM1_08_BEST_Split_1{1,iKolomDiskrit}(1,2) ;
        % ------------------------------
        % Kalau data AWAL <= BEST SPLIT
        % ------------------------------
        if dataAwal <= bestSplit
            CM1_09_Biner_1(iBarisDiskrit,iKolomDiskrit) = 0;            
        % -----------------------------
        % Kalau data AWAL > BEST SPLIT
        % -----------------------------
        else CM1_09_Biner_1(iBarisDiskrit,iKolomDiskrit) = 1;
        end
        
        % --------------------------------------------
        % Menambahkan kolom kelas ke "CM1_09_Biner_1"
        % --------------------------------------------
        if iKolomDiskrit == 21 %Fitur ke 21                          
            dataKelasnya = CM1_04_DataFitur{1,iKolomDiskrit}(iBarisDiskrit,2); % ambil data kelas dari "CM1_04_DataFitur"
            CM1_09_Biner_1(iBarisDiskrit,iKolomDiskrit+1) = dataKelasnya; % data kelas disimpan di kolom ke 22
        end                                    
    end           
end   
        
% ----------------------------------------------------------------------
% Distinct data MtrainingBiner_1 --> agar tidak ada redudansi data biner
% ----------------------------------------------------------------------
CM1_10_Biner_1_Unique = unique(CM1_09_Biner_1,'rows'); % Data redundan diseleksi (include kelas)

% ------------------------------------------------------------------------------------------------------------------------------
% Jika jumlah "Mtraining05UniqueBiner_1" DENGAN dan TANPA kelas itu berbeda, pasti ada duplikasi data dengan kelas yang berbeda
% ------------------------------------------------------------------------------------------------------------------------------
uniqueDenganKelas = length(CM1_10_Biner_1_Unique); % jumlah unique dengan kelasnya juga
uniqueTanpaKelas = length(unique(CM1_10_Biner_1_Unique(:,1:21),'rows')); % Data unique tanpa kelas
if  uniqueTanpaKelas ~= uniqueDenganKelas % Data unique tanpa kelas ~= data unique 
%---

    % ----------------------------------------------------------------------------
    % Perbandingan jumlah unique DENGAN dan TANPA kelas di "CM1_10_Biner_1_Unique"
    % ----------------------------------------------------------------------------
    CM1_11_Perbandingan_1 = [uniqueTanpaKelas uniqueDenganKelas uniqueDenganKelas-uniqueTanpaKelas]; 
        
%==================================================================================================================================================================================
%                                              ********************** EBD FASE 2 **********************
%==================================================================================================================================================================================
    
    % ----------------------------------------------------------------------------------------------------
    % Pembagian data "CM1_07_DataFiturASC_Split_Distinct" berdasarkan "CM1_08_BEST_Split_1" menjadi 2A dan 2B
    % ----------------------------------------------------------------------------------------------------
    for ikolomFoldSplit = 1 : 21        
        A = 1; % Karena counter A dan B berbeda
        B = 1; % Karena counter A dan B berbeda
        jmlSplitDist = size(CM1_07_DataFiturASC_Split_Distinct{1,ikolomFoldSplit},1); % Pake size, bukan length, 3 row x 9 col maka length muncul 9                     
        for iBarisData = 1 : jmlSplitDist            
            bestSplit_1 = CM1_08_BEST_Split_1{1,ikolomFoldSplit}(1,2);
            dataSplit_1 = CM1_07_DataFiturASC_Split_Distinct{1,ikolomFoldSplit}(iBarisData,1);            
            if dataSplit_1 <= bestSplit_1 
                CM1_12_Split_1_2A{1,ikolomFoldSplit}(A,1) = dataSplit_1;
                A = A + 1;
            else
                CM1_13_Split_1_2B{1,ikolomFoldSplit}(B,1) = dataSplit_1;                
                B = B + 1;
            end
        end
    end        
        
    % -------------------------------
    % Split data "CM1_12_FiturASC_2A"
    % -------------------------------
    for iKolomFold2A = 1 : 21                               
        jumlah2A = size(CM1_12_Split_1_2A{1,iKolomFold2A},1); % Banyaknya data di "CM1_12_FiturASC_2A"            
        nilaiBestSplit = CM1_08_BEST_Split_1{1,iKolomFold2A}(1,2); % Nilai best splitnya berapa
        % ---------------------------------------------
        % Jumlah "CM1_12_Split_1_2A" 1, ga perlu split
        % ---------------------------------------------
        if jumlah2A == 1                
            CM1_14_Split_1_2A_Split{1,iKolomFold2A}(1,1) = nilaiBestSplit;                
        % -----------------------------------------
        % Kalau lebih dari satu datanya, siap split
        % -----------------------------------------
        else            
            for iBaris2A = 1 : jumlah2A - 1 % Dikurangi satu                                                                                                                
                % ----------------------------------------
                % Urutan data yang terakhir tidak di-split
                % ----------------------------------------
                dataPertama = CM1_12_Split_1_2A{1,iKolomFold2A}(iBaris2A,1); % Urutan data split pertama
                dataKedua = CM1_12_Split_1_2A{1,iKolomFold2A}(iBaris2A+1,1); % Urutan data split kedua
                hasilSplit2A = (dataPertama+dataKedua)/2; % Ditambah dan dibagi dua, nilainya disimpan di kolom 1                                             
                CM1_14_Split_1_2A_Split{1,iKolomFold2A}(iBaris2A,1) = hasilSplit2A;                                        
            end
        end                            
    end
        
    % -------------------------------
    % Split data "Mtraining06Urut_2B"
    % -------------------------------        
    for iKolomFold2B = 1 : 21         
        % -------------------------------------------
        % Antisipasi, kalau "CM1_13_FiturASC_2B" = []
        % -------------------------------------------
        if size(CM1_13_Split_1_2B{1,iKolomFold2B},1) ~= 0                   
        %--                   
            jumlah2B = size(CM1_13_Split_1_2B{1,iKolomFold2B},1); % Banyaknya data di "CM1_13_FiturASC_2B"            
            nilai2B = CM1_13_Split_1_2B{1,iKolomFold2B}(1,1); % Nilai 2B satu-satunya            
            % --------------------------------------------------------
            % Jumlah data "CM1_13_FiturASC_2B" cuma 1, ga perlu split
            % --------------------------------------------------------
            if jumlah2B == 1                   
                CM1_15_Split_1_2B_Split{1,iKolomFold2B}(1,1) = nilai2B;                
            % -----------------------------------------
            % Kalau lebih dari satu datanya, siap split
            % -----------------------------------------
            else                
                for iBaris2B = 1 : jumlah2B - 1 % Dikurangi satu                                                                                                                    
                    % ----------------------------------------
                    % Urutan data yang terakhir tidak di-split
                    % ----------------------------------------
                    dataPertama = CM1_13_Split_1_2B{1,iKolomFold2B}(iBaris2B,1); % Urutan data split pertama
                    dataKedua = CM1_13_Split_1_2B{1,iKolomFold2B}(iBaris2B+1,1); % Urutan data split kedua
                    hasilSplit2B = (dataPertama+dataKedua)/2; % Ditambah dan dibagi dua, nilainya disimpan di kolom 1                                             
                    CM1_15_Split_1_2B_Split{1,iKolomFold2B}(iBaris2B,1) = hasilSplit2B;                                        
                end
            end                
        %--    
        end                                        
    end     
        
    % ---------------------------------------------------------------------------------------------------------------------------
    % Pembagian data "CM1_04_DataFitur" berdasarkan "CM1_08_BEST_Split_1" -> Jadi "CM1_16_DataFitur_2A" dan "CM1_17_DataFitur_2B"
    % ---------------------------------------------------------------------------------------------------------------------------
    for iKolomTraining = 1 : 21        
        A = 1;
        B = 1;
        for iBarisTraining = 1 : jmlData
            numerikAwal = CM1_04_DataFitur{1,iKolomTraining}(iBarisTraining,1);
            dataBestSplit = CM1_08_BEST_Split_1{1,iKolomTraining}(1,2);
            dataKelasnya = CM1_04_DataFitur{1,iKolomTraining}(iBarisTraining,2);
            % ------------------------------------
            % Kalau <= best split maka training 2A
            % ------------------------------------
            if numerikAwal <= dataBestSplit                
                CM1_16_DataFitur_2A{1,iKolomTraining}(A,:) = [numerikAwal dataKelasnya];
                A = A + 1;
            % ------------------------------------
            % Kalau > best split maka training 2B
            % ------------------------------------
            else                
                CM1_17_DataFitur_2B{1,iKolomTraining}(B,:) = [numerikAwal dataKelasnya];
                B = B + 1;
            end
        end
    end
    
    % -------------------------------------------------------------------
%1  % Update kolom pada "CM1_14_Split_1_2A_Split" dengan data training 2A
    % -------------------------------------------------------------------
    % Jumlah TRUE ( <= ) data split FASE 2A          [2] 
    % Jumlah FALSE ( <= ) data split FASE 2A         [3] 
    % Entropy CHILDREN ( <= ) di data split FASE 2A  [4] 
    % Jumlah TRUE ( > ) data split FASE 2A           [5] 
    % Jumlah FALSE ( > ) data split FASE 2A          [6] 
    % Entropy CHILDREN ( > ) di data split FASE 2A   [7] 
    % Nilai INFO dari setiap data split 2A           [8] 
    % Nilai GAIN dari setiap data split 2A           [9]         
    % ------------------------------------------------------------
%2  % Mencari nilai GAIN (max) dari setiap FITUR:
    % "CM1_18_Best_Split_2A" --> [barisKe,angkaSplit,nilaiGain]            
    % ------------------------------------------------------------
    fase_2A_fix;        
 
    % -------------------------------------------------------------------
%4  % Update kolom pada "CM1_15_Split_1_2B_Split" dengan data training 2B
    % -------------------------------------------------------------------
    % Jumlah TRUE ( <= ) data split FASE 2B          [2] 
    % Jumlah FALSE ( <= ) data split FASE 2B         [3] 
    % Entropy CHILDREN ( <= ) di data split FASE 2B  [4] 
    % Jumlah TRUE ( > ) data split FASE 2B           [5] 
    % Jumlah FALSE ( > ) data split FASE 2B          [6] 
    % Entropy CHILDREN ( > ) di data split FASE 2B   [7] 
    % Nilai INFO dari setiap data split 2B           [8] 
    % Nilai GAIN dari setiap data split 2B           [9]           
    % ------------------------------------------------------------
%5  % Mencari nilai GAIN (max) dari setiap FOLD dan FITUR:
    % "CM1_19_Best_Split_2B" --> [barisKe,angkaSplit,nilaiGain]                                
    % ------------------------------------------------------------
    fase_2B_fix;                         
    
    % -----------------------------------------------------------------------
    % Transformasi data NUMERIK ke HEXA berdasarkan BEST split 1, 2A, dan 2B
    % -----------------------------------------------------------------------
    % Alasan konversi ke HEXA, agar bisa menghilangkan redundansi
    % Kalau ke string, ketika diimplementasi fungsi unique datanya jadi 4
    % saja yaitu 00, 01, 10, dan 11
    % Kalau ke biner, sulit, karena ada cell di dalam cell, jadi perlu effort lebih menghilangkan redundansi
    
        
    % ---------------
    % NUMERIK to HEXA
    % ---------------
    for iFitur = 1 : 21
        for iBaris = 1 : jmlData % Banyaknya data training               
            
            trainingSekarang = CM1_04_DataFitur{1,iFitur}(iBaris,1);
            kelasnya = CM1_04_DataFitur{1,iFitur}(iBaris,2);                    
            split1 = CM1_08_BEST_Split_1{1,iFitur}(1,2);
            split2A = CM1_18_BEST_Split_2A{1,iFitur}(1,2);
                    
            % -----------------------------------------------------------------
            % Mencegah ambil best split 2B kalau memang nilainya tidak ada = []
            % -----------------------------------------------------------------
            if length(CM1_19_BEST_Split_2B{1, iFitur}) ~= 0                                    
                split2B = CM1_19_BEST_Split_2B{1, iFitur}(1,2);      
            end    
                
            %------------------------------------------------------------------------------
            % Proses BINING, kalau "data numerik awal" <= "BEST split 1" maka termasuk "2A"
            %------------------------------------------------------------------------------
            if trainingSekarang <= split1 % <= split 1
                % -----------
                % <= split 2A
                % -----------
                if trainingSekarang <= split2A 
                    CM1_20_Hexa_2(iBaris,iFitur) = 0;
                % ----------
                % > split 2A
                % ----------
                else CM1_20_Hexa_2(iBaris,iFitur) = 1;
                end  
            %--------------------------------------------------------------
            % Kalau "data numerik awal" > "BEST split 1" maka termasuk "2B"
            %--------------------------------------------------------------
            else % > split 1
                % -----------
                % <= split 2B
                % -----------
                if trainingSekarang <= split2B                      
                    CM1_20_Hexa_2(iBaris,iFitur) = 2;
                % ----------
                % > split 2B
                % ----------
                else CM1_20_Hexa_2(iBaris,iFitur) = 3;                               
                end                             
            end 
                
            % --------------------------------------------
            % Menambahkan kolom kelas pada "CM1_20_Hexa_2"
            % --------------------------------------------
            if iFitur == 21 % nambahin kelas                                      
                CM1_20_Hexa_2(iBaris,22) = kelasnya;
                CM1_20_Hexa_2(iBaris,23) = iBaris;                    
            end                 
        end % iFitur                
    end % iBaris                        
                                                                                                                                                                            
    % -------------------------------------
    % Remove redundansi HEXA incldude kelas
    % -------------------------------------
    CM1_21_Hexa_2_Unique = unique(CM1_20_Hexa_2(:,1:22),'rows');      
                            
    % ---------------------------------------------------------
    % Cari perbandingan Unique HEX dengan kelas DAN tanpa kelas
    % ---------------------------------------------------------
    uniqueHEXdenganKelas = length(CM1_21_Hexa_2_Unique); % jumlah unique dengan kelasnya juga
    uniqueHEXtanpaKelas = length(unique(CM1_21_Hexa_2_Unique(:,1:21),'rows')); % Data unique tanpa kelas
    if  uniqueHEXdenganKelas ~= uniqueHEXtanpaKelas % Data unique tanpa kelas ~= data unique                         
        
        % ----------------------------------------------------------------------------
        % Perbandingan jumlah unique DENGAN dan TANPA kelas di "CM1_10_Biner_1_Unique"
        % ----------------------------------------------------------------------------
        CM1_22_Perbandingan_2 = [uniqueHEXtanpaKelas uniqueHEXdenganKelas uniqueHEXdenganKelas-uniqueHEXtanpaKelas];
        
        % ----------------------------------------------------------------------
        % Transformasi data training (NUMERIK) ke BINER dan HEXA dari EBD 2 FASE
        % ----------------------------------------------------------------------
        for iKolomFold = 1 : 21
            for iBaris = 1 : jmlData % Banyaknya data training
                for iFitur = 1 : 21  
                    
                    trainingSekarang = CM1_04_DataFitur{1,iFitur}(iBaris,1);
                    kelasnya = CM1_04_DataFitur{1,iFitur}(iBaris,2);                    
                    split1 = CM1_08_BEST_Split_1{1,iFitur}(1,2);
                    split2A = CM1_18_BEST_Split_2A{1,iFitur}(1,2);
                    
                    % -----------------------------------------------------------------
                    % Mencegah ambil best split 2B kalau memang nilainya tidak ada = []
                    % -----------------------------------------------------------------
                    if length(CM1_19_BEST_Split_2B{1, iFitur}) ~= 0
                        split2B = CM1_19_BEST_Split_2B{1, iFitur}(1,2);      
                    end                                                
                    
                    % ---------------------------------------------------
                    % apakah kolom yang ingin dituju? (dijadikan 2 digit)
                    % ---------------------------------------------------
                    if iFitur == iKolomFold 
                        if trainingSekarang <= split1 % <= split 1                            
                            if trainingSekarang <= split2A % <= split 2A                                                                                                   
                                CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,iFitur) = 0;
                            else % > split 2A                                                                                                
                                CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,iFitur) = 1;
                            end                            
                        else % > split 1
                            if trainingSekarang <= split2B % <= split 2B
                                CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,iFitur) = 2;
                            else % > split 2B
                                CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,iFitur) = 3;
                            end                             
                        end 
                        if iFitur == 21 % nambahin kelas                                                        
                            CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,22) = kelasnya;
                            CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,23) = iBaris;                            
                        end 
                    else % Bukan fitur yang dituju
                        if trainingSekarang <= split1 % <= split 1                                                        
                            CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,iFitur) = 0;
                        else % > split 1
                            CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,iFitur) = 1;                            
                        end                        
                        if iFitur == 21 % nambahin kelas                            
                            CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,22) = kelasnya;
                            CM1_23_Hexa_2_PerFitur{1,iKolomFold}(iBaris,23) = iBaris;                            
                        end                        
                    end % iFitur == iKolomFold                    
                end % iFitur                
            end % iBaris                        
            
            % ------------------------------------------------
            % Remove redundansi biner di EBD 2 FASE pake kelas
            % ------------------------------------------------
            CM1_24_Hexa_2_PerFitur_Unique{1,iKolomFold} = unique(CM1_23_Hexa_2_PerFitur{1,iKolomFold}(:,1:22),'rows');                      
            
            % ---------------------------------------------------------
            % Cari perbandingan Unique HEX dengan kelas DAN tanpa kelas
            % ---------------------------------------------------------
            uniqueHEXfiturDenganKelas = length(CM1_24_Hexa_2_PerFitur_Unique{1,iKolomFold}); % jumlah unique dengan kelasnya juga
            uniqueHEXfiturTanpaKelas = length(unique(CM1_24_Hexa_2_PerFitur_Unique{1,iKolomFold}(:,1:21),'rows')); % Data unique tanpa kelas
            if  uniqueHEXfiturDenganKelas ~= uniqueHEXfiturTanpaKelas % Data unique tanpa kelas ~= data unique                  
                selisih = uniqueHEXfiturDenganKelas - uniqueHEXfiturTanpaKelas;
                CM1_25_Perbandingan_2_PerFitur{1,iKolomFold} = [uniqueHEXfiturTanpaKelas uniqueHEXfiturDenganKelas selisih];  
                kumpulanSelisih(iKolomFold,1) = selisih;                
            else
                disp('Tidak ada redundansi EBD Fase 2 per Fitur!');
            end                         
        end %iKolomFold    
        
        % -------------------------------------------------------------------------
        % Di sini harusnya ada split 3 FASE dst. hingga tidak ada lagi over-vitting
        % -------------------------------------------------------------------------
        %
        %

    else disp('Tidak ada redundansi EBD Fase 2!');
    end                
    
else disp('Tidak ada redundansi EBD Fase 1!');
%---
end

% ------------------------------------------------------------
% Hitung rata-rata selisih di "CM1_25_Perbandingan_2_PerFitur"
% ------------------------------------------------------------
CM1_26_Average_Selisih_PerFitur = mean(kumpulanSelisih);
   
clear iJumlahTF jmlFalse jmlTrue jmlData entropyParent iDataPecah iFiturPecah;
clear dataPertama dataKedua iDataSplit iFiturSplit;
clear dataASC dataSplit iBarisASC iBarisSplit iKolomUpdate;
clear jmlFalseKurang jmlFalseLebih jmlTrueKurang jmlTrueLebih kelasASC;
clear entropyChildKurang piFalseKurang piTrueKurang totalKurang;
clear entropyChildLebih piFalseLebih piTrueLebih totalLebih;
clear angkaBestSplit BarisKe dataChildKurang dataChildLebih GAINinfo INFOsplit Nilai;
clear bestSplit dataAwal dataKelasnya iBarisDiskrit iKolomDiskrit;
clear uniqueDenganKelas uniqueTanpaKelas;
clear A B bestSplit_1 dataSplit_1 iBarisData ikolomFoldSplit jmlSplitDist;
clear hasilSplit2A iBaris2A iKolomFold2A jumlah2A nilaiBestSplit;
clear hasilSplit2B iBaris2B iKolomFold2B jumlah2B nilai2B;
clear dataBestSplit iBarisTraining iKolomTraining numerikAwal;
clear angkaSplitA BarisKeA dataChildKurangA dataChildLebihA dataKelasA dataSplitA dataTrainingA;
clear entropyChildKurangA entropyChildLebihA GAINinfoA iBarisSplitA iBarisTrainingA iKolomCellA INFOsplitA;
clear jmlFalseKurangA jmlFalseLebihA jmlTrueKurangA jmlTrueLebihA NilaiA panjangSplit2A panjangTraining2A;
clear piFalseKurangA piFalseLebihA piTrueKurangA piTrueLebihA totalKurangA totalLebihA;
clear angkaSplitB BarisKeB dataChildKurangB dataChildLebihB dataKelasB dataSplitB dataTrainingB;
clear entropyChildKurangB entropyChildLebihB GAINinfoB iBarisSplitB iBarisTrainingB iKolomCellB INFOsplitB;
clear jmlFalseKurangB jmlFalseLebihB jmlTrueKurangB jmlTrueLebihB NilaiB panjangSplit2B panjangTraining2B;
clear piFalseKurangB piFalseLebihB piTrueKurangB piTrueLebihB totalKurangB totalLebihB;
clear iBaris iFitur iKolomCell kelasnya split1 split2A split2B trainingSekarang;
clear uniqueHEXdenganKelas uniqueHEXtanpaKelas;
clear iKolomFold uniqueHEXfiturDenganKelas uniqueHEXfiturTanpaKelas selisih kumpulanSelisih;

toc

disp('Saving...');
    tic
        save('02_EBD\CM1_fix.mat');        
    toc
disp('Done!');