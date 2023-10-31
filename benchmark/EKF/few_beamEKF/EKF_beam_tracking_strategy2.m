clear all;
close all;
clc

% relative delta t (每100個一個訓練通道，總長1000)
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


% % for loops
% file_size = 256;
% file_number = 10;  % 要做幾個檔案********
% file_count = 0;
t_length = 1000; % 總長度 ************
% % record normalized beamforming gains
beam_loss_sery_total = zeros(t_length, 6);
few_beam_number = 11;
strategy = 2 ;
speed_count = 0;

% first loop for different speeds
for v = 5 : 5 : 30
% for v = 30
    a = v * 0.2;
    % read files 
    file_name = ['data_few_v' num2str(v) '_a' num2str(a) '.0_beam' num2str(few_beam_number) '_Strategy' num2str(strategy) '_ODE_3CNN_1LSTM_60epoch.mat'];
    load(file_name);
    [UE_num, time_slots] = size(label);
    speed_count = speed_count + 1;
    beam_loss_sery = zeros(t_length, 1);
    
    for j = 1:UE_num
        % record normalized beamforming gains
        % mmWave received signals of beam training  64beam
        mm_signal_sery = squeeze(channel(j, 1, :, :) + 1i * channel(j, 2, :, :));
        % mmWave channels 64beam
        mm_channel_sery = mm_signal_sery / candidate_narrow_beam;  % 在產生資料時已經有做beamforming，現在還原而已
        training_num = 0;
        for t_count = 1 : time_slots-1
            % consider the t_count time slot
            mm_channel = squeeze(mm_channel_sery(t_count, :));
            if(mod(t_count, 100) == 1) % beam training的時間點
                training_num = training_num+1;
                start = ext(j,training_num,1)+1;  % python的資料從0開始
                last = ext(j,training_num,2)+1;  
                channel_train = mm_channel(1,start:last);
                signal_few = squeeze(channel_few(j, 1, training_num, :) + 1i * channel_few(j, 2, training_num, :));
                signal_few  = signal_few';  % 轉成跟channel一樣維度
                % MUSIC
                alpha = - MUSIC(channel_train,size(channel_train,2));
                % record the best beam
                [received_signal, max_beam] = max(signal_few);  % 有被干擾的答案  已更改編號
                temp = [1,start:last];
                true_beam = temp(1,max_beam);
                beam_vector = candidate_narrow_beam(:, true_beam);  % 抓出他的beam power
                beam_vector_few = beam_vector(start:last,1);
                % calculate the pathloss and channel coefficients
                origin_coeff = beam_vector_few.' * exp(-1i * pi * sin(alpha) * [0 : few_beam_number - 1])';
                received_signal = received_signal / origin_coeff;
                normalized_received_signal = received_signal / abs(received_signal);
                pathloss = abs(received_signal);
                s = [real(normalized_received_signal), imag(normalized_received_signal), alpha, 0, 0];% state initialization
                % calculate normalized beamforming gain  對答案~~~~
                beam_loss_sery(t_count) = beam_loss_sery(t_count) + (beam_power(j, t_count, max_beam) / ...
                    max(beam_power(j, t_count, :)))^2;
                % initialize covariance matrix
                U = Q;
            else  % 這邊開始做predict
                % otherwise: EKF
                % state update
                s = (F * s')';
                % predicted received signal
                h = beam_vector_few.' * exp(-1i * pi * sin(s(3)) * [0 : few_beam_number - 1])' * (s(1) + 1i * s(2)) * pathloss;
                
                % gradient matrix
                G = zeros(length(s), 1);
                G(1) = beam_vector_few.' * exp(-1i * pi * sin(s(3)) * [0 : few_beam_number - 1])' * pathloss;
                G(2) = 1i * beam_vector_few.' * exp(-1i * pi * sin(s(3)) * [0 : few_beam_number - 1])' * pathloss;
                dh = beam_vector_few.' * exp(-1i * pi * sin(s(3) + d) * [0 : few_beam_number - 1])' * (s(1) + 1i * s(2)) * pathloss - h;
                G(3) = dh / d;
                G = G.';
                
                % true received signal
                h_true = signal_few(1, max_beam);
                
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
    beam_loss_sery_total(:, speed_count) = beam_loss_sery_total(:, speed_count) + beam_loss_sery / UE_num;
end

% final results
beam_loss_sery_total = reshape(beam_loss_sery_total , 100, 10, 6); %*************
avg_result = mean(mean(beam_loss_sery_total(2 : end, :, :), 2), 1);
fileName = ['EKF_result_beam_' num2str(few_beam_number) '.mat'];
save(fileName, 'beam_loss_sery_total');






