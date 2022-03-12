clear; close all; clc       % Tidy up

%% Declare audio and experiment parameters
audio_length = 180;                 % Length of recording [s]
Fs = 10e3;                          % Sampling frequency [Hz]
Tint = 5;                           % Interval of eyes open/closed [s]
num_samples = audio_length * Fs;    % Number of samples from data

signal = audioread("05_08_2021.wav", [1 num_samples]);
signal = signal/max(abs(signal));   % Normalising signal
time = linspace(0, audio_length, num_samples);

%% Plot voltage signal

plot(time, signal);
for n=0:Tint:time(end)
    xline(n, ':')   % Plot lines for Tint second segments
end

xlabel('Time (s)')
ylabel('Measured Electrode Voltage (µV)')
title('Measured Electrode Voltage vs. Time for eyes closed and open data')
legend('Electrode Voltage', 'Segments')
set(gca, 'FontSize', 18)
axis([xlim [0 max(ylim)]])
% axis square
axis tight

%% Get "one-sided" FFT amplitude and power spectra

Fsig = fft(signal) / length(signal);
aFsig = abs(Fsig);     % get amplitude of fft

p_spectrum = aFsig.^2; % two-sided power spectrum
p_spectrum = p_spectrum(1:floor(length(Fsig)/2));
p_spectrum(2:end) = 2*p_spectrum(2:end); % one-sided power spectrum

aFsig = aFsig(1:floor(length(Fsig)/2));
aFsig(2:end) = 2*aFsig(2:end);

%% Plotting "one-sided" FFT amplitude and power spectra

max_freq = 40;  % Max frequency to plot spectra for
freqs = 1/audio_length * (0:(length(p_spectrum)));
idx = floor(max_freq/(Fs/2/length(freqs)));

figure; hold on
plot(freqs(1:idx), aFsig(1:idx), '-ob')
xlabel('Frequency (Hz)')
ylabel('FFT amplitude')
set(gca, 'FontSize', 18)
axis([xlim 0 max(ylim)])
axis square
axis tight

P0 = p_spectrum(1); % DC component as the reference component
p_spectrum_db = 10*log10(p_spectrum / P0);

figure; hold on
plot(freqs(1:idx), p_spectrum_db(1:idx), '-or')
xlabel('Frequency (Hz)')
ylabel('Power (dB rel. to DC)')
set(gca, 'FontSize', 18)
axis([xlim 0 max(ylim)])
axis square
axis tight
