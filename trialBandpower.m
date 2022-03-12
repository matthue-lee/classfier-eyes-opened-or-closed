function [pow_a,pow_b] = trialBandpower(trial_data, pow_ref)
%trialBandpower Computes the average power in the alpha band (8-12 Hz) and 
%beta band (16-30 Hz) for a 5-second long signal or single trial of EEG
%Inputs:
%   trial_data = column vector containing 5-second long segment of EEG
%                corresponding to one trial where subject's eyes were
%                either opened or closed
%   pow_ref = reference power given by DC component of FFT on the raw 
%             voltage signal
%Outputs:
%   pow_a = scalar floating point number given by power of the input signal
%           in the alpha band (8-12 Hz)
%   pow_b = scalar floating point number given by power of the input signal
%           in the beta band (16-30 Hz)


% Get "one-sided" Power spectrum
Fsig = fft(trial_data) / length(trial_data);
aFsig = abs(Fsig);     % get amplitude of fft

p_spectrum = aFsig.^2; % two-sided power spectrum
p_spectrum = p_spectrum(1:floor(length(Fsig)/2));

p_spectrum(2:end) = 2*p_spectrum(2:end); % one-sided power spectrum
p_spectrum = 10*log10(p_spectrum / pow_ref);

trial_length = 5;   % Interval of eyes open/closed [s]
freqs = 1/trial_length * (0:(length(p_spectrum)));

alpha_band = (freqs >= 8 & freqs <= 12);
beta_band = (freqs >= 16 & freqs <= 30);

pow_a = mean(p_spectrum(alpha_band));
pow_b = mean(p_spectrum(beta_band));

end

