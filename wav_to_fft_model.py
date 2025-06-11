import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import spectrogram

# Sample rate (CD-quality)
fs = 44100  

# Duration for each note
duration = 1.0  

# Frequencies of C4 and E4
freq_C4 = 261.63  
freq_E4 = 220

# Generate time arrays
t1 = np.linspace(0, duration, int(fs * duration), endpoint=False)
t2 = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate sine waves for each note
note_C4 = 0.5 * np.sin(2 * np.pi * freq_C4 * t1)
note_E4 = 0.5 * np.sin(2 * np.pi * freq_E4 * t2)

# Concatenate the two notes
audio = np.concatenate((note_C4, note_E4))

# Save as WAV
write("C4_E4.wav", fs, (audio * 32767).astype(np.int16))  # Save as 16-bit PCM

# Analyze with Spectrogram
f, t, Sxx = spectrogram(audio, fs, window='blackman', nperseg=4096, noverlap=512)

plt.figure(figsize=(10, 4))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=-70, vmax=-40)
plt.colorbar(label='dB')
plt.title("Spectrogram of C4 followed by E4")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.ylim(0, 500)  # Focus on piano note range
plt.axhline(freq_C4, color='red', linestyle='--', label='C4')
plt.axhline(freq_E4, color='green', linestyle='--', label='E4')
plt.legend()
plt.tight_layout()
plt.show()

# find peaks
time_target = 1.5
time_index = np.argmin(np.abs(t - time_target))

spectrum_slice = Sxx[:, time_index]

top_n = 3
top_indices = np.argsort(spectrum_slice)[-top_n:][::-1]  # descending order

dominant_freqs = f[top_indices]
dominant_powers = spectrum_slice[top_indices]

for freq, power in zip(dominant_freqs, dominant_powers):
    print(f"Frequency: {freq:.2f} Hz, Power: {10*np.log10(power):.2f} dB")

from scipy.signal import find_peaks

peaks, _ = find_peaks(spectrum_slice, height=np.max(spectrum_slice)*0.1)
peak_freqs = f[peaks]

print("Peaks found at frequencies:", peak_freqs)