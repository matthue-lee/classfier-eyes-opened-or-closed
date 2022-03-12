clear; close all; clc       % Tidy up

%% Declare audio and experiment parameters
audio_length = 180;                 % Length of recording [s]
Fs = 10e3;                          % Sampling frequency [Hz]
Tint = 5;                           % Interval of eyes open/closed [s]
num_samples = audio_length * Fs;    % Number of samples from data

signal = audioread("05_08_2021.wav", [1 num_samples]);

%% Getting reference DC component from power spectrum

Fsig = fft(signal) / length(signal);
p_spectrum = abs(Fsig).^2; % two-sided power spectrum
pow_ref = p_spectrum(1);   % DC component as the reference component    

%% Compute alpha and beta bandpower for each of the 36 trials

num_trials = audio_length/Tint;
trial_length = length(signal)/num_trials;
time = linspace(0, Tint, trial_length);

pow_a = zeros(num_trials, 1);
pow_b = zeros(num_trials, 1);

for n=1:num_trials
    % Get bandpower for each of the 36 trials
    nstart = (n-1)*trial_length+1;
    nstop = n*trial_length;
    
    trial_data = signal(nstart:nstop);
    [pow_a(n), pow_b(n)] = trialBandpower(trial_data, pow_ref);
end

%% Plotting bandpower of each trial

states = ["Eyes closed", "Eyes open"];
groups = repmat(states, 1, num_trials/length(states))';

figure
gscatter(pow_a, pow_b, groups, 'br', 'o')
xlabel('Mean alpha band power (dB/Hz rel. DC)')
ylabel('Mean beta band power (dB/Hz rel. DC)')
title("Mean power for Alpha vs. Beta bands");

% Setting axis limits
ax_range = round([min(pow_a) max(pow_a) min(pow_b) max(pow_b)]);
ax_range([1 3]) = ax_range([1 3]) - 1;
ax_range([2 4]) = ax_range([2 4]) + 1;
axis(ax_range)

%% Find cross-validated accuracy

num_held = 2;
num_folds = num_trials / num_held;
correct = zeros(num_folds, 1);

% Get start indices of each held out fold for testing
start_idx = (num_trials - num_held + 1):-num_held:1;

% Create feature matrix containing bandpowers
features = [pow_a pow_b];

% Specify fold number to plot
fold_to_plot = 17;

%Collect all correct across all folds
total_num_correct = 0;

for k = 1:num_folds
    % Get indices of all held out and held in trials
    held_out = start_idx(k):start_idx(k)+num_held-1;
    held_in=[1:(start_idx(k)-1) start_idx(k)+num_held:num_trials];
    
    % Using LDA classifier to discriminate between the two states
    [class,~,~,~,coeff] = classify(features(held_out,:), ...
                                   features(held_in, :), ...
                                   groups(held_in), 'linear');
    
    actual_states = groups(held_out);
    
    % Store number of correctly classified trials in each fold
    correct(k) = sum(actual_states==class)
    
    % Store required variables for plotting
    if (k == fold_to_plot)
        held_out_plt = held_out;
        held_in_plt = held_in;
        coeff_plt = coeff;
    end
end

% Get proportional of correctly classified trials
num_folds

num_trials

total_correct = sum(correct)

% Get Decoding Accuracy (DA) as a percentage
DA = total_correct/num_trials * 100

%% Plotting bandpower for held-in and held-out trials
figure
hold on

gscatter(pow_a(held_out_plt),pow_b(held_out_plt),groups(held_out_plt),'br')
gscatter(pow_a(held_in_plt),pow_b(held_in_plt),groups(held_in_plt),'br','o')

% Plotting linear decision boundary for one fold
K = coeff_plt(1,2).const;
L = coeff_plt(1,2).linear; 
f = @(x,y) K+L(1)*x + L(2)*y;
h = fimplicit(f);
set(h,'DisplayName','Decision Boundary', 'LineStyle','--', 'Color', 'k');

xlabel('Mean alpha band power (dB/Hz rel. DC)')
ylabel('Mean beta band power (dB/Hz rel. DC)')
title("Mean power for Alpha vs. Beta bands (LDA Classifier)");
axis(ax_range)


%% Getting indices of two consecutive pairs of each state
trial_num = 3;          % Trial number for pair (1 to 18)

n = trial_num*2 - 1;    % Get trial number for "eyes closed" state
nstart = (n-1)*trial_length+1;
nstop = n*trial_length;

closed_trial = signal(nstart:nstop);
open_trial = signal((nstart:nstop) + trial_length);

%% Plotting timeseries for two consecutive pairs of each state

figure
subplot(2,1,1)

plot(time, closed_trial)
yline(rms(closed_trial), 'r--')
xlabel('Time (s)')
ylabel('Voltage (µV)')
title('Electrode voltage vs. time for eyes closed segment')
legend('Electrode voltage', 'RMS voltage')
axis([0 Tint min(signal) max(signal)])

subplot(2,1,2)
plot(time, open_trial)
yline(rms(open_trial), 'r--')
xlabel('Time (s)')
ylabel('Voltage (µV)')
title('Electrode voltage vs. time for eyes open segment')
legend('Electrode voltage', 'RMS voltage')
axis([0 Tint min(signal) max(signal)])

%% Plotting PSD for two consecutive pairs of each state

% Getting power spectral density of "eyes closed" trial in decibels
Fclosed = fft(closed_trial) / length(closed_trial);
Pclosed = abs(Fclosed).^2; % two-sided power spectrum
Pclosed = Pclosed(1:floor(length(Fclosed)/2));
Pclosed(2:end) = 2*Pclosed(2:end); % one-sided power spectrum
Pclosed = 10*log10(Pclosed / Pclosed(1));

% Getting power spectral density of "eyes open" trial in decibels
Fopen = fft(open_trial) / length(open_trial);
Popen = abs(Fopen).^2; % two-sided power spectrum
Popen = Popen(1:floor(length(Fopen)/2));
Popen(2:end) = 2*Popen(2:end); % one-sided power spectrum
Popen = 10*log10(Popen / Popen(1));

max_freq = 35;  % Max frequency to plot spectra for
freqs = 1/Tint * (0:(length(Pclosed)));
idx = floor(max_freq/(Fs/2/length(freqs)))+1;

figure; hold on
plot(freqs(1:idx), Pclosed(1:idx))
plot(freqs(1:idx), Popen(1:idx))

xlabel('Frequency (Hz)')
ylabel('Power (dB rel. to DC)')
title('Power Spectral Density for Eyes Closed and Eyes Open trial')

alpha_band = polyshape([8 8 12 12],[-15 35 35 -15]);
plot(alpha_band)
beta_band = polyshape([16 16 30 30],[-15 35 35 -15]);
plot(beta_band)

axis tight
axis([0 max_freq -15 35])
set(gca, 'FontSize', 14) 
legend([states, 'Alpha band', 'Beta band'], 'FontSize', 10)
