import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

# === Load audio ===
y, sr = librosa.load("SampleAudioWav\Do.wav", sr=None)

# === CQT Parameters ===
hop_length = 512
n_bins = 84            # 7 octaves (e.g., C1–C8)
bins_per_octave = 12   # Semitones

# === Compute CQT ===
C = librosa.cqt(y, sr=sr, hop_length=hop_length,
                n_bins=n_bins, bins_per_octave=bins_per_octave)

# === Convert amplitude to dB ===
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)


class PitchDetector(nn.Module):
    def __init__(self, n_layers, n_input_bins, n_notes):
        super().__init__()

        n_intermediate_layers = n_layers - 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_input_bins, 256))
        for i in range(n_intermediate_layers):
            self.layers.append(nn.Linear(256, 256))
        self.layers.append(nn.Linear(256, n_notes))

        '''# encoder layers
        step_enc = (num_question - k) // enc_layers
        self.encoder = nn.ModuleList()
        prev = num_question
        for i in range(enc_layers):
            next_size = num_question - (i + 1) * step_enc
            if i == enc_layers - 1:
                next_size = k
            self.encoder.append(nn.Linear(prev, next_size))
            prev = next_size'''

    def get_weight_norm(self):
        return sum(torch.norm(m.weight, 2) ** 2
                   for m in self.modules()
                   if isinstance(m, nn.Linear))

    def forward(self, inputs):
        
        out = inputs
        for layer in self.layers:
            out = torch.sigmoid(layer(out))

        return out
