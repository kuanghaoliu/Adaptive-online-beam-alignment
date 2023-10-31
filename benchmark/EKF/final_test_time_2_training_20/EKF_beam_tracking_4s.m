clear all;
close all;
clc

% relative delta t (0.001秒採樣一次)
dt = 0.001;

% state transition matrix
F = [1, 0, 0, 0, 0;...
     0, 1, 0, 0, 0;...
     0, 0, 1, dt, dt^2 / 2;...
     0, 0, 0, 1, dt;...
     0, 0, 0, 0, 1];
% covariance matrix
Q = diag([1 - 0.95^2, 1 - 0.95^2, 0.1 / 2 * dt ^ 2, 0.1 * dt, dt]);
% epsilon
d = 1e-8;
% constant
C = 1e-4;

% MM antenna num
MM_narrow_beam_antenna_num = 64;
% MM narrow beam num
MM_narrow_beam_num = 64;
% angular range

sector_start = - pi;
sector_end = pi;
% narrow beam generation
candidate_narrow_beam_angle = sector_start + (sector_end - sector_start) / MM_narrow_beam_num * [0.5 : 1 : MM_narrow_beam_num - 0.5];
candidate_narrow_beam = exp(-1i * [0 : MM_narrow_beam_antenna_num - 1]' * candidate_narrow_beam_angle) / sqrt(MM_narrow_beam_num);

% for loops
file_size = 256;
file_number = 10;  % 要做幾個檔案********
file_count = 0;
t_length = 4000; % 總長度 ************
% record normalized beamforming gains
beam_loss_sery_total = zeros(t_length, 6);

speed_count = 0;
% first loop for different speeds
for v = 5 : 5 : 30

    speed_count = speed_count + 1;
    beam_loss_sery = zeros(t_length, 1);
    % second loop for different files
    for i = 1 + file_count : file_number + file_count
        % read files 
        file_name = ['C:\Users\Aiden\Desktop\my_data\time_01_SNR_104\final_4s\ODE_dataset(final_test)_v' num2str(v) '\dataset_v' num2str(v) '_' num2str(i) '.mat'];
        load(file_name);
        % record normalized beamforming gains
        % third loop for different samples
        for j = 1 : file_size
            % mmWave received signals of beam training  算出訊號Y
            mm_signal_sery = squeeze(MM_data(j, 1, :, :) + 1i * MM_data(j, 2, :, :));
            % mmWave channels
            mm_channel_sery = mm_signal_sery / candidate_narrow_beam;
            % for each time slot  # 每個時間點
            for t_count = 1 : t_length
                % consider the t_count time slot
                mm_channel = squeeze(mm_channel_sery(t_count, :));
                mm_signal = squeeze(mm_signal_sery(t_count, :));


                
                if(mod(t_count, 100) == 1) % beam training的時間點
                    % MUSIC
                    alpha = - MUSIC(mm_channel);
                    % record the best beam
                    [received_signal, max_beam] = max(mm_signal);
                    beam_vector = candidate_narrow_beam(:, max_beam);
                    % calculate the pathloss and channel coefficients
                    origin_coeff = beam_vector.' * exp(-1i * pi * sin(alpha) * [0 : MM_narrow_beam_antenna_num - 1])';
                    received_signal = received_signal / origin_coeff;
                    normalized_received_signal = received_signal / abs(received_signal);
                    pathloss = abs(received_signal);
                    s = [real(normalized_received_signal), imag(normalized_received_signal), alpha, 0, 0];% state initialization
                    % calculate normalized beamforming gain
                    beam_loss_sery(t_count) = beam_loss_sery(t_count) + (beam_power(j, t_count, max_beam) / ...
                        max(beam_power(j, t_count, :)))^2;
                    % initialize covariance matrix
                    U = Q;
                else  % 這邊開始做predict
                    % otherwise: EKF
                    % state update
                    s = (F * s')';
                    % predicted received signal
                    h = beam_vector.' * exp(-1i * pi * sin(s(3)) * [0 : MM_narrow_beam_antenna_num - 1])' * (s(1) + 1i * s(2)) * pathloss;
                    
                    % gradient matrix
                    G = zeros(length(s), 1);
                    G(1) = beam_vector.' * exp(-1i * pi * sin(s(3)) * [0 : MM_narrow_beam_antenna_num - 1])' * pathloss;
                    G(2) = 1i * beam_vector.' * exp(-1i * pi * sin(s(3)) * [0 : MM_narrow_beam_antenna_num - 1])' * pathloss;
                    dh = beam_vector.' * exp(-1i * pi * sin(s(3) + d) * [0 : MM_narrow_beam_antenna_num - 1])' * (s(1) + 1i * s(2)) * pathloss - h;
                    G(3) = dh / d;
                    G = G.';
                    
                    % true received signalˋㄍˋ
                    h_true = mm_signal_sery(t_count, max_beam);
                    
                    % update priori convariance matrix
                    U = F * U * F' + Q;
                    
                    % Kalman filter
                    K = U * G' / (C + G * U * G');
                    
                    % update posterior coefficients
                    s = s + K.' * (h_true - h);
                    U = (1 - K.' * G.') * U;
                    s = real(s);
                    
                    % calculate normalized beamformin gain
                    predicted_angle = sin(s(3));
                    [~, predicted_id] = min(abs(candidate_narrow_beam_angle / pi - predicted_angle));
                    beam_loss_sery(t_count) = beam_loss_sery(t_count) + (beam_power(j, t_count, predicted_id) / ...
                        max(beam_power(j, t_count, :)))^2;
                end
            end
        end
    end
    beam_loss_sery_total(:, speed_count) = beam_loss_sery_total(:, speed_count) + beam_loss_sery / file_size;
end

% final results
beam_loss_sery_total = reshape(beam_loss_sery_total / file_number, 100, 40, 6); %*************
avg_result = mean(mean(beam_loss_sery_total(2 : end, :, :), 2), 1)
save('EKF_result.mat', 'beam_loss_sery_total');