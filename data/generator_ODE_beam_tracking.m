clear all;
close all;
clc;

% ODE beam tracking
% DeepMIMO, O1, only consider BS1
% 28GHz
% row: 100-900

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

% UE distribution
row_index = [100 : 900];
% load and save MM channel into MM_ch, row by row
MM_ch = zeros(length(row_index), 181, 64);
count = 1;
for i = row_index
    MM_file = ['C:\Users\Aiden\Desktop\my_data\DeepMIMOv2\DeepMIMO_dataset\MM_DeepMIMO_dataset_' num2str(i) '_row.mat'];
    load(MM_file);
    % beam training results
    MM_ch(count, :, :) = squeeze(MM_channel(1, :, :)) * candidate_narrow_beam;
    count = count + 1;
end

% file number: 40 for training and 10 for testing
file_num = 50;
% sample number in each file
file_size = 256;
% UE speed
speeds = 5 : 5 : 30;

% UE acceleration
a_max = speeds * 0.2;

% MM beam training received signal
MM_data = zeros(file_size, 2, 101, MM_narrow_beam_num);
% MM optimal beam index
beam_label = zeros(file_size, 101);
% MM beam amplitude
beam_power = zeros(file_size, 101, MM_narrow_beam_num);

% for different UE speeds
for speed = speeds
    % make dictionary
    mkdir(['ODE_dataset_v' num2str(speed)]);
    for i = 1 : file_num
        for j = 1 : file_size
            % find UE trajectory within the pre-defined range
            flag = 0;
            while flag == 0
                initial_x = round(200 + rand * 600);
                initial_y = round(rand * 181);
                direction = rand * 2 * pi;
                a = rand * speed * 0.2;
                % beam tracking duration: 1.6 s
                % beam training period: 0.16 s
                % beam prediction resolution: 0.016 s
                location = round([initial_x, initial_y] + (speed / 0.2 * [0 : 0.01 : 1]' + 0.5 * a / 0.2 * ([0 : 0.01 : 1] .^ 2)')* [cos(direction), sin(direction)]);
                if min(location(:, 1)) >= 100 && max(location(:, 1)) <= 900 && ...
                    min(location(:, 2)) >= 1 && max(location(:, 2)) <= 181
                    flag = 1;
                end
            end
            % save corresponding data
            % MM_data: sequence of mmWave beam training received signal
            % vector
            % beam_label: sequence of mmWave optimal beam
            % beam_power: sequence of mmWave beam training received signal
            % power vector
            for k = 1 : 101
                [~, beam_label(j, k)] = max(squeeze(MM_ch(location(k, 1) - 99,location(k, 2), :)));
                MM_data(j, 1, k, :) = real(MM_ch(location(k, 1) - 99,location(k, 2), :));
                MM_data(j, 2, k, :) = imag(MM_ch(location(k, 1) - 99,location(k, 2), :));
                beam_power(j, k, :) = abs(MM_ch(location(k, 1) - 99,location(k, 2), :));
            end
        end
        MM_data = awgn(MM_data, 104);
        save(['ODE_dataset_v' num2str(speed) '\dataset_v' num2str(speed) '_' num2str(i) '.mat'], ...
            'MM_data', 'beam_label', 'beam_power');
    end

end