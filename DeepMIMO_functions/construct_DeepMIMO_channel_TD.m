% --------- DeepMIMO: A Generic Dataset for mmWave and massive MIMO ------%
% Authors: Ahmed Alkhateeb, Umut Demirhan, Abdelrahman Taha 
% Date: March 17, 2022
% Goal: Encouraging research on ML/DL for MIMO applications and
% providing a benchmarking tool for the developed algorithms
% ---------------------------------------------------------------------- %

function [channel]=construct_DeepMIMO_channel_TD(tx_ant_size, tx_rotation, tx_ant_spacing, rx_ant_size, rx_rotation, rx_ant_spacing, params_user, params)

ang_conv=pi/180;

% TX antenna parameters for a UPA structure
M_TX_ind = antenna_channel_map(tx_ant_size(1), tx_ant_size(2), tx_ant_size(3), 0);
M_TX=prod(tx_ant_size);
kd_TX=2*pi*tx_ant_spacing;

% RX antenna parameters for a UPA structure
M_RX_ind = antenna_channel_map(rx_ant_size(1), rx_ant_size(2), rx_ant_size(3), 0);
M_RX=prod(rx_ant_size);
kd_RX=2*pi*rx_ant_spacing;

if params_user.num_paths == 0
    channel = complex(zeros(M_RX, M_TX, params_user.num_paths));
    return
end

% Change the DoD and DoA angles based on the panel orientations
if params.activate_array_rotation
    [DoD_theta, DoD_phi, DoA_theta, DoA_phi] = axes_rotation(tx_rotation, params_user.DoD_theta, params_user.DoD_phi, rx_rotation, params_user.DoA_theta, params_user.DoA_phi);
else
    DoD_theta = params_user.DoD_theta;
    DoD_phi = params_user.DoD_phi;
    DoA_theta = params_user.DoA_theta;
    DoA_phi = params_user.DoA_phi;
end

% Apply the radiation pattern of choice
if params.radiation_pattern % Half-wave dipole radiation pattern
    power = params_user.power.* antenna_pattern_halfwavedipole(DoD_theta, DoD_phi) .* antenna_pattern_halfwavedipole(DoA_theta, DoA_phi);
else % Isotropic radiation pattern
    power = params_user.power;
end


% TX Array Response - BS
gamma_TX=+1j*kd_TX*[sind(DoD_theta).*cosd(DoD_phi);
              sind(DoD_theta).*sind(DoD_phi);
              cosd(DoD_theta)];
array_response_TX = exp(M_TX_ind*gamma_TX);

% RX Array Response - UE or BS
gamma_RX=+1j*kd_RX*[sind(DoA_theta).*cosd(DoA_phi);
                    sind(DoA_theta).*sind(DoA_phi);
                    cosd(DoA_theta)];
array_response_RX = exp(M_RX_ind*gamma_RX);

%Generate the time domain (TD) impulse response where each 2D slice represents the TD channel matrix for one specific path delay
path_const=sqrt(power).*exp(1j*params_user.phase*ang_conv);
channel = reshape(array_response_RX, M_RX, 1, []) .* reshape(array_response_TX, 1, M_TX, []) .* reshape(path_const, 1, 1, params_user.num_paths);

end