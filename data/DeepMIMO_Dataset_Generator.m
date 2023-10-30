% ----------------- Add the path of DeepMIMO function --------------------%
addpath('DeepMIMO_functions')

% -------------------- DeepMIMO Dataset Generation -----------------------%

% total row number of UE distribution
row_num = 2751;
% UE number in each row
UE_row_num = 181;
% mmWave BS number
BS_num = 12;
% candidate mmWave channel
MM_channel = zeros(BS_num, UE_row_num, 64);

% mmWave channel parameters
params = read_params('parameters.m');
% for each row
for i = 1 : row_num
    % select active users (this row)
    params.active_user_first = i;
    params.active_user_last = i;
    print_info = ['mmWave dataset ' num2str(i) 'th row generation started'];
    % generate corresponding channels and parameters
    [dataset_MM, params_MM] = DeepMIMO_generator(params);
    % save channels into matrices
    for j = 1 : BS_num
        for k = 1 : UE_row_num
            MM_channel(j, k, :) = squeeze(sum(dataset_MM{j}.user{k}.channel, 3));
        end
    end
    % save channels into files
    %fprintf('\n Saving the DeepMIMO Dataset ...')
    sfile_DeepMIMO = ['C:\Users\Aiden\Desktop\my_data\DeepMIMOv2\DeepMIMO_dataset/MM_DeepMIMO_dataset_' num2str(i) '_row.mat'];
    save(sfile_DeepMIMO,'MM_channel', '-v7.3');
end



% -------------------------- Output Examples -----------------------------%
% DeepMIMO_dataset{i}.user{j}.channel % Channel between BS i - User j
% %  (# of User antennas) x (# of BS antennas) x (# of OFDM subcarriers)
%
% DeepMIMO_dataset{i}.user{j}.params % Parameters of the channel (paths)
% DeepMIMO_dataset{i}.user{j}.LoS_status % Indicator of LoS path existence
% %     | 1: LoS exists | 0: NLoS only | -1: No paths (Blockage)|
%
% DeepMIMO_dataset{i}.user{j}.loc % Location of User j
% DeepMIMO_dataset{i}.loc % Location of BS i
%
% % BS-BS channels are generated only if (params.enable_BSchannels == 1):
% DeepMIMO_dataset{i}.basestation{j}.channel % Channel between BS i - BS j
% DeepMIMO_dataset{i}.basestation{j}.loc
% DeepMIMO_dataset{i}.basestation{j}.LoS_status
%
% % Recall that the size of the channel vector was given by 
% % (# of User antennas) x (# of BS antennas) x (# of OFDM subcarriers)
% % Each of the first two channel matrix dimensions follows a certain 
% % reshaping sequence that can be obtained by the following
% % 'antennamap' vector: Each entry is 3 integers in the form of 
% % 'xyz' where each representing the antenna number in x, y, z directions
% antennamap = antenna_channel_map(params.num_ant_BS(1), ...
%                                  params.num_ant_BS(2), ...
%                                  params.num_ant_BS(3), 1);
%
% -------------------------- Dynamic Scenario ----------------------------%
%
% DeepMIMO_dataset{f}{i}.user{j}.channel % Scene f - BS i - User j
% % Every other command applies as before with the addition of scene ID
% params{f} % Parameters of Scene f
%
% ------------------------------------------------------------------------%