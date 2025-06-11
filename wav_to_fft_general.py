import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import spectrogram

fs, audio = read("project-one\SampleAudioWav\Ditto arpeg.wav")

# Analyze with Spectrogram
f, t, Sxx = spectrogram(audio, fs, window='blackman', nperseg=4096, noverlap=1024)

plt.figure(figsize=(10, 4))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=-70, vmax=50)
plt.colorbar(label='dB')
plt.title("Spectrogram")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.ylim(0, 2000)  # Focus on piano note range
plt.legend()
plt.tight_layout()
plt.show()

# find peaks
time_target = 2.5 # Adjust this to the desired time in seconds
time_index = np.argmin(np.abs(t - time_target))
print(t[time_index])

spectrum_slice = Sxx[:, time_index]

top_n = 10
top_indices = np.argsort(spectrum_slice)[-top_n:][::-1]  # descending order

dominant_freqs = f[top_indices]
dominant_powers = spectrum_slice[top_indices]

for freq, power in zip(dominant_freqs, dominant_powers):
    print(f"Frequency: {freq:.2f} Hz, Power: {10*np.log10(power):.2f} dB")

from scipy.signal import find_peaks

peaks, _ = find_peaks(spectrum_slice, height=np.max(spectrum_slice)*0.1)
peak_freqs = f[peaks]

print("Peaks found at frequencies:", peak_freqs)

# Simple HPS sketch
from scipy.fft import fft
from math import ceil

#frame = audio[:4096] * np.hanning(4096)
#spectrum = np.abs(fft(frame))[:2048]

spectrum = spectrum_slice

spec_len = len(spectrum)

hps = spectrum.copy()
for h in range(2, 5):
    print(h)
    print(len(spectrum[::h]), len(hps[:ceil(spec_len / h)]))
    hps[:ceil(spec_len / h)] *= spectrum[::h]

fundamental_index = np.argmax(hps)
fundamental_freq = fundamental_index * fs / 4096
print(f"Estimated fundamental frequency: {fundamental_freq:.2f} Hz")