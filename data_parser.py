import librosa
import librosa.display
import numpy as np
import pandas as pd

import pretty_midi

import os

csv_metadata_path = "MAESTRO Data\maestro-v1.0.0\maestro-v1.0.0.csv"

def load_dataset(n_samples=None, shuffle=False, delta_t=0.02):
    metadata_df = pd.read_csv(csv_metadata_path)
    metadata_array = metadata_df.values  # to np array

    print(metadata_array)

    if shuffle:
        np.random.shuffle(metadata_array)
        print(metadata_array)

    if n_samples is not None:
        metadata_array = metadata_array[:n_samples]

    all_data, all_labels = [], []
    i = 0
    for sample in metadata_array:
        i += 1
        audio_path = os.path.normpath("MAESTRO Data/maestro-v1.0.0/" + sample[5])
        midi_path = os.path.normpath("MAESTRO Data/maestro-v1.0.0/" + sample[4])

        _, sr = load_audio(audio_path)

        # CQT params
        hop_length = int(sr * delta_t) # 512 or power of 2 for comp efficiency
        n_bins = 88 # 88 keys on piano
        bins_per_octave = 12 # semitones
        fmin = librosa.note_to_hz('A0')

        CQT_data, sr = load_data(audio_path, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin)

        if CQT_data is None or sr is None:
            continue
        
        labels = load_labels(midi_path, sr, hop_length, total_frames=CQT_data.shape[0], n_bins=n_bins, fmin=fmin)

        if labels is None:
            continue

        # shape match check
        if CQT_data.shape[0] != labels.shape[0]:
            print(f"Warning: Shape mismatch! Features: {CQT_data.shape}, Labels: {labels.shape}")
            min_length = min(CQT_data.shape[0], labels.shape[0])
            CQT_data = CQT_data[:min_length]
            labels = labels[:min_length]
        
        all_data.append(CQT_data)
        all_labels.append(labels)
        
        print(f"Added: {CQT_data.shape} features, {labels.shape} labels")

    return all_data, all_labels

    #train_ratio = 0.7
    #val_ratio = 0.15

    # Split into train/validation/test
    #return split_data(all_data, all_labels, train_ratio, val_ratio)


def split_data(all_data, all_labels, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets"""
    total_samples = len(all_data)
    
    # Calculate split indices
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Split the data
    train_data = all_data[:train_end]
    train_labels = all_labels[:train_end]
    
    valid_data = all_data[train_end:val_end]
    valid_labels = all_labels[train_end:val_end]
    
    test_data = all_data[val_end:]
    test_labels = all_labels[val_end:]
    
    print(f"Split: {len(train_data)} train, {len(valid_data)} validation, {len(test_data)} test")
    
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels


def load_data(audio_path, hop_length=512, n_bins=88, bins_per_octave=12, fmin=27.5):
    y, sr = load_audio(audio_path)

    if y is None or sr is None:
        return None, None

    # CQT
    C = librosa.cqt(y, sr=sr, hop_length=hop_length,
                    n_bins=n_bins, bins_per_octave=bins_per_octave,
                    fmin=fmin)

    # to dB
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

    CQT_data = C_db.T # time x n_bins
    return CQT_data, sr


def load_labels(midi_path, sr, hop_length, total_frames, n_bins=88, fmin=27.5):
    mid = pretty_midi.PrettyMIDI(midi_path)

    if mid is None:
        return None
    
    # Create empty label matrix: (n_frames, n_bins)
    labels = np.zeros((total_frames, n_bins), dtype=int)
    
    # For each instrument in the MIDI file
    for instrument in mid.instruments:
        for note in instrument.notes:
            # Convert note to frequency bin
            pitch = note.pitch
            # Piano notes: 21 (A0) to 108 (C8)
            # Our bins: 0-87 corresponding to A0-C8
            bin_idx = pitch - 21  # Convert MIDI note number to bin index
            
            if 0 <= bin_idx < n_bins:  # Ensure it's within piano range
                # Convert time to frame indices
                start_frame = int(note.start * sr / hop_length)
                end_frame = int(note.end * sr / hop_length)
                
                # Ensure frames are within bounds
                start_frame = max(0, min(start_frame, total_frames - 1))
                end_frame = max(0, min(end_frame, total_frames - 1))
                
                # Set labels to 1 for active note duration
                labels[start_frame:end_frame, bin_idx] = 1
    
    return labels


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


#train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_dataset(n_samples=2)

#all_data = train_data + valid_data + test_data
#all_labels = train_labels + valid_labels + test_labels

all_data, all_labels = load_dataset(n_samples=2)

X = np.concatenate(all_data, axis=0)    # Shape: (total_frames, 88)
y = np.concatenate(all_labels, axis=0)  # Shape: (total_frames, 88)

print(f"All data samples: {X.shape}")
print(f"All labels samples: {y.shape}")
