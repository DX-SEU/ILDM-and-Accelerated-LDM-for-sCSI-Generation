clc;clear;
addpath quadriga_src\
Nr = 128;               % Number of received antennas
Nv = 128;               % Number of valid subscarriers
Ng = 144;               % Number of cyclic prefix
Nc = 2048;              % Number of all subscarriers
fc = 4.8e9;             % Carrier frequency (Hz)
df = 15e3;              % Frequency space
BW = df * Nc;
%% Create general settings
para_sim = qd_simulation_parameters;
para_sim.center_frequency = fc;             % Set center frequency
para_sim.use_random_initial_phase = 0;      % Disable random initial phase
para_sim.show_progress_bars = 0;            % Disable progress bar
para_sim.sample_density = para_sim.speed_of_light / 2 / fc;
%% qd_layout
para_layout = qd_layout(para_sim);                                      % Create a new qd_layout object
para_layout.set_scenario('3GPP_38.901_UMa_NLOS');                       % Set scenario
para_layout.rx_array = qd_arrayant.generate('3gpp-3d', 1, Nr, fc, 1);   % Set the receive antenna (BS)
para_layout.rx_position = [0, 0, 25]';                                  % Receiver plsition
para_layout.tx_array = qd_arrayant.generate('omni');                    % Set the transmit antenna (UT)
numSum = 500;                % The number of expectation for statistal channel acquisition
space = 1;                  % Grid space
grid_len = 100;             % Length of target area
grid_center = [200, 0];     % Center coordinate of the target area
xtick = 0:space:grid_len;
ytick = 0:space:grid_len;
tx_pos = gen_grid(xtick, ytick, numSum);
numGrid = (grid_len / space + 1) ^ 2;                                   % Total number of the grid points
para_layout.tx_track.set_scenario('3GPP_38.901_UMa_NLOS')               % Track scenario
para_layout.tx_track.set_speed(1e-10)                                   % Small enough to approach static state
para_layout.tx_track.initial_position = [grid_center(1)-grid_len/2, -grid_len/2, 1.5]';
para_layout.tx_track.positions = [para_layout.tx_track.positions, tx_pos];
para_layout.tx_track.no_segments = para_layout.tx_track.no_snapshots;   % Use spatial consisteny for mobility
para_layout.visualize([],[],0)
axis equal
%% Generate channel
% These parameters can be adjustable
cb = para_layout.init_builder;
numCluster = 30;
cb.scenpar.DS_mu = -7.2;
cb.scenpar.DS_sigma = 0.5;
cb.scenpar.SF_sigma = 0;
cb.scenpar.NumClusters = numCluster;
cb.scenpar.NumSubPaths = 1;
cb.scenpar.SC_lambda = 1;
cb.scenpar.PerClusterDS = 0;
cb.gen_parameters;
channel = cb.get_channels;
channel.individual_delays = 0;
% Generate SF and AD domai channel
G_tmp = squeeze(channel.fr(BW, Nv));
ADCPMGrid = zeros(numGrid, Nr, Nv);
cnt = 1;
for i = 1:numGrid
    ADCPMTemp = zeros(Nr, Nv);
    for j = 1:numSum
        cnt = cnt + 1;
        h_ad = ifft2(G_tmp(:, :, cnt));
        ADCPMTemp = ADCPMTemp + h_ad .* conj(h_ad);
    end
    ADCPMGrid(i, :, :) = ADCPMTemp;
end
ADCPMGrid = ADCPMGrid / numSum;
coff_norm = sqrt( mean( mean(abs(ADCPMGrid).^2, 2), 3) );
ADCPMGrid = ADCPMGrid ./ repmat(coff_norm, [1, Nr, Nv]);        % ADCPM Omega
save('ADCPMGrid.mat', 'ADCPMGrid', 'ADCPMGrid', '-v7.3');
gridLoc = zeros(2, length(xtick)*length(ytick));
a_temp = repmat(xtick + grid_center(1) - grid_len / 2, length(xtick), 1);
gridLoc(1, :) = a_temp(:)';
gridLoc(2, :) = repmat(ytick + grid_center(2) - grid_len / 2, 1, length(ytick));
gridLoc = gridLoc';
save('gridLoc.mat', 'gridLoc', 'gridLoc', '-v7.3');
% Dataset division
idx = randperm(size(gridLoc, 1));
proportion_of_train = 0.9;
numTrain = ceil(size(gridLoc, 1) * proportion_of_train);
numTest = size(gridLoc, 1) - numTrain;
ADCPMTrain = ADCPMGrid(idx(1:numTrain), :, :);
trainLoc = gridLoc(idx(1:numTrain), :);
ADCPMTest = ADCPMGrid(idx(numTrain+1:end), :, :);
testLoc = gridLoc(idx(numTrain+1:end), :);
save('data/VAE_dataset/ADCPMTrain.mat', 'ADCPMTrain', 'ADCPMTrain', '-v7.3');
save('data/VAE_dataset/ADCPMTest.mat', 'ADCPMTest', 'ADCPMTest', '-v7.3');
save('data/ILDM_dataset/trainLoc.mat', 'trainLoc', 'trainLoc', '-v7.3');
save('data/ILDM_dataset/testLoc.mat', 'testLoc', 'testLoc', '-v7.3');
