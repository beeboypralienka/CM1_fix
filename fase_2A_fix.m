% Input:
%--------
% Data "CM1_14_Split_1_2A_Split" -> Data split 2A
% Data "CM1_16_DataFitur_2A"  -> Data training untuk 2A

% ----------------------------------------------------------------------------------------------------------
% Cari jumlah TRUE dan FALSE serta nilai ENTROPY children di Mtraining berdasarkan "CM1_14_Split_1_2A_Split"
% ----------------------------------------------------------------------------------------------------------
    
for iKolomCellA = 1 : 21 % Iterasi fitur CM1 ada 21 (exclude kelas)       
    %--      
    panjangSplit2A = length(CM1_14_Split_1_2A_Split{1,iKolomCellA}); % Setiap DATA SPLIT diulang sebanyak jumlah DATA TRAINING
    panjangTraining2A = size(CM1_16_DataFitur_2A{1,iKolomCellA},1); % Iterasi data training AGAR MATCH dengan SATU DATA split    
    for iBarisSplitA = 1 : panjangSplit2A  % data SPLIT 2A       
        
        % ----------------------------------------------------------------------
        % Di-NOL-kan, karena jumlah TRUE dan FALSE setiap data split itu berbeda
        % ----------------------------------------------------------------------                
        jmlTrueKurangA = 0;
        jmlFalseKurangA = 0;
        jmlTrueLebihA = 0;
        jmlFalseLebihA = 0;
        
        for iBarisTrainingA = 1 : panjangTraining2A % data TRAINING 2A            
            % -----------------------------------------------------------
            % Hitung jumlah TRUE dan FALSE dari kategoti ( <= ) dan ( > )
            % -----------------------------------------------------------                            
            dataTrainingA = CM1_16_DataFitur_2A{1, iKolomCellA}(iBarisTrainingA,1); % Data training
            dataKelasA = CM1_16_DataFitur_2A{1, iKolomCellA}(iBarisTrainingA,2); % Data kelas                          
            dataSplitA = CM1_14_Split_1_2A_Split{1, iKolomCellA}(iBarisSplitA,1); % Data split                                        
            if dataTrainingA <= dataSplitA % ada berapa data training yang ( <= ) data split                   
                if  dataKelasA == 1 % Hitung jumlah TRUE pada parameter ( <= )                          
                    jmlTrueKurangA = jmlTrueKurangA + 1; % Hitung jumlah TRUE ( <= )                         
                else % Hitung jumlah FALSE pada parameter ( <= )                    
                    jmlFalseKurangA = jmlFalseKurangA + 1; % Hitung jumlah FALSE ( <= )
                end
            else % ada berapa data training yang ( > ) data split                        
                if dataKelasA == 1 % Hitung jumlah TRUE dan FALSE pada parameter ( > )                                                
                    jmlTrueLebihA = jmlTrueLebihA + 1; % Hitung jumlah TRUE ( > )                        
                else                            
                    jmlFalseLebihA = jmlFalseLebihA + 1; % Hitung jumlah FALSE ( > )
                end
            end
        end
        % ---------------------------
        % Update kolom 2, 3, 5, dan 6
        % ---------------------------
        CM1_14_Split_1_2A_Split{1,iKolomCellA}(iBarisSplitA,2) = jmlTrueKurangA; % Jumlah TRUE dengan parameter ( <= ) disimpan di kolom 2
        CM1_14_Split_1_2A_Split{1,iKolomCellA}(iBarisSplitA,3) = jmlFalseKurangA; % Jumlah FALSE dengan parameter ( <= ) disimpan di kolom 3
        CM1_14_Split_1_2A_Split{1,iKolomCellA}(iBarisSplitA,5) = jmlTrueLebihA; % Jumlah TRUE dengan parameter ( > ) disimpan di kolom 5
        CM1_14_Split_1_2A_Split{1,iKolomCellA}(iBarisSplitA,6) = jmlFalseLebihA; % Jumlah FALSE dengan parameter ( > ) disimpan di kolom 6                                

        % ---------------------------------------------
        % Cari entropy child "2A" dari parameter ( <= )
        % ---------------------------------------------                       
        totalKurangA = jmlTrueKurangA + jmlFalseKurangA; % Total jumlah TRUE dan jumlah FALSE dari parameter ( <= )              
        if totalKurangA ~=0 % Selama total jumlah TRUE dan FALSE bukan NOL pada parameter ( <= )                                
            piTrueKurangA(iBarisSplitA,1) = jmlTrueKurangA / (jmlTrueKurangA+jmlFalseKurangA); % Hitung jumlah TRUE ( <= )
            piFalseKurangA(iBarisSplitA,1) = jmlFalseKurangA / (jmlTrueKurangA+jmlFalseKurangA); % Hitung jumlah FALSE ( <= )                
            if piTrueKurangA(iBarisSplitA,1) == 0 || piFalseKurangA(iBarisSplitA,1) == 0 % Jika hasil Pi TRUE atau Pi FALSE itu NOL, dipastikan entropyChild (<=) juga NOL                                        
                entropyChildKurangA(iBarisSplitA,1) = 0; % Entropy child ( <= ) dijadikan NOL
            else % Jika hasil ( <= ) Pi TRUE dan Pi FALSE bukan NOL                                                            
                % ----------------------------
                % Hitung entropy child ( <= )
                % ----------------------------
                entropyChildKurangA = entropyChildrenEBD_fix(piTrueKurangA,piFalseKurangA,iBarisSplitA);
            end                
        else % Jika total jumlah TRUE dan FALSE adalah NOL pada parameter ( <= ), maka dipastikan entropyChild (<=) juga NOL                    
                    entropyChildKurangA(iBarisSplitA,1) = 0; % Entropy child ( <= ) dijadikan NOL
        end             
        CM1_14_Split_1_2A_Split{1,iKolomCellA}(iBarisSplitA,4) = entropyChildKurangA(iBarisSplitA,1); % Nilai entropy child dari parameter ( <= ) disimpan di kolom 4                          

        % --------------------------------------------
        % Cari entropy child "2A" dari parameter ( > )
        % --------------------------------------------                         
        totalLebihA = jmlTrueLebihA + jmlFalseLebihA; % Total jumlah TRUE dan jumlah FALSE dari parameter ( > )                        
        if totalLebihA ~= 0 % Selama total jumlah TRUE dan FALSE bukan NOL pada parameter ( > )            
            piTrueLebihA(iBarisSplitA,1) = jmlTrueLebihA / (jmlTrueLebihA+jmlFalseLebihA); % Hitung jumlah TRUE ( > )
            piFalseLebihA(iBarisSplitA,1) = jmlFalseLebihA / (jmlTrueLebihA+jmlFalseLebihA); % Hitung jumlah FALSE ( > )                
            if piTrueLebihA(iBarisSplitA,1) == 0 || piFalseLebihA(iBarisSplitA,1) == 0 % Jika hasil Pi TRUE atau Pi FALSE itu NOL, dipastikan entropyChild ( > ) juga NOL                
                entropyChildLebihA(iBarisSplitA,1) = 0; % Entropy child ( > ) dijadikan NOL
            else % Jika hasil ( > ) Pi TRUE dan Pi FALSE bukan NOL                                        
                % ---------------------------
                % Hitung entropy child ( > )
                % ---------------------------                    
                entropyChildLebihA = entropyChildrenEBD_fix(piTrueLebihA, piFalseLebihA,iBarisSplitA);                   
            end
        else % Jika total jumlah TRUE dan FALSE adalah NOL pada parameter ( > )                
            entropyChildLebihA(iBarisSplitA,1) = 0; % Entropy child ( > ) dijadikan NOL
        end          
        CM1_14_Split_1_2A_Split{1,iKolomCellA}(iBarisSplitA,7) = entropyChildLebihA(iBarisSplitA,1); % Nilai entropy child dari parameter ( > ) disimpan di kolom 7 

        % -----------------------------------------
        % Mencari nilai INFO dari setiap data split
        % -----------------------------------------
        dataChildKurangA = (totalKurangA/jmlData) * CM1_14_Split_1_2A_Split{1, iKolomCellA}(iBarisSplitA,4);
        dataChildLebihA = (totalLebihA/jmlData) * CM1_14_Split_1_2A_Split{1, iKolomCellA}(iBarisSplitA,7);
        INFOsplitA(iBarisSplitA,1) = (dataChildKurangA + dataChildLebihA);
        CM1_14_Split_1_2A_Split{1,iKolomCellA}(iBarisSplitA,8) = INFOsplitA(iBarisSplitA,1); % nilai INFO dari data SPLIT. disimpan di kolom 8

        % ------------------------------------
        % Mencari nilai GAIN dari setiap INFO
        % ------------------------------------
        GAINinfoA(iBarisSplitA,1) = CM1_03_Keterangan(1,4) - INFOsplitA(iBarisSplitA,1);
        CM1_14_Split_1_2A_Split{1,iKolomCellA}(iBarisSplitA,9) = GAINinfoA(iBarisSplitA,1); % nilai INFO dari data SPLIT. disimpan di kolom 9                        

        % ----------------------------------------------------------------------------------------------------------------------------
        % Penyederhanaan variable "CM1_14_Split_1_2A_Split" 
        % [1] Data Split, [2] TRUE(<=), [3] FALSE(<=), [4] entropy(<=), [5] TRUE(>), [6] FALSE(>), [7] entropy(>), [8] INFO, [9] GAIN
        % ----------------------------------------------------------------------------------------------------------------------------                
    end     
    % ---------------------------------------------------------------
    % Mencari nilai best split berdasarkan nilai GAIN tertinggi (max)
    % ---------------------------------------------------------------
    [NilaiA,BarisKeA] = max(CM1_14_Split_1_2A_Split{1,iKolomCellA}(:,9)); % Ambil urutan ke berapa si split terbaik itu dan ambil nilai max gain-nya
    angkaSplitA = CM1_14_Split_1_2A_Split{1, iKolomCellA}(BarisKeA,1); % Angka split terbaik dari daftar urut split
    CM1_18_BEST_Split_2A{1,iKolomCellA} = [BarisKeA angkaSplitA NilaiA]; % nilai max Gain dari data split ke berapa                
%--
end        
        
% Output:
% --------
% Entropy children A ( <= ) dan ( > )
% Nilai INFO dari setiap data split di FITUR dan FOLD tertentu
% Nilai GAIN dari setiap INFO di FITUR dan FOLD tertentu
% "CM1_14_Split_1_2A_Split" dengan total 9 kolom
% Nilai Gain (MAX) sebagai split terbaik "CM1_18_BEST_Split_2A"