import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch

import data_parser as dp
from nn_models import PitchCNN

def label_to_notes(labels):
    """
    Convert a binary array of length 88 (piano keys) into note names (A0–C8).
    """
    # sanity check
    if len(labels) != 88:
        raise ValueError("Input array must have length 88 for piano keys A0–C8")

    # generate note names
    names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    notes = []
    octave = 0
    note_index = names.index('A')  # piano starts on A0
    for i in range(88):
        note_name = names[note_index % 12] + str(octave)
        notes.append(note_name)
        note_index += 1
        if names[note_index % 12] == 'C':  # bump octave when hitting C
            octave += 1

    # map binary → notes
    pressed = [notes[i] for i, bit in enumerate(labels) if bit == 1]

    return pressed

def bin_to_note_name(bin_index):
    """
    Convert bin index (0-87) to piano note name (A0-C8)
    """
    # Piano note names (12 notes per octave)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate MIDI note number (A0 = 21, C8 = 108)
    midi_note = bin_index + 21
    
    # Calculate octave and note index
    octave = (midi_note // 12) - 1
    note_index = midi_note % 12
    
    # Get note name
    note_name = note_names[note_index]
    
    return f"{note_name}{octave}"

def create_prediction_windows(features, context_size=4):
    """
    Convert frame-level features into overlapping time windows for prediction.
    This version doesn't check labels and handles edge cases with padding.

    features: (time, freq_bins)
    context_size: number of frames per window
    """
    X = []
    half = context_size // 2
    T = features.shape[0]
    
    # Handle edge cases by padding or adjusting window boundaries
    for t in range(T):
        # Calculate window boundaries
        start = t - half
        end = t + half + 1
        
        # Handle beginning of sequence (pad with first frame)
        if start < 0:
            padding_needed = -start
            window_start = 0
            # Pad by repeating the first frame
            pad_frames = np.tile(features[0:1], (padding_needed, 1))  # Repeat first frame
            window_content = features[window_start:end]
            window = np.vstack([pad_frames, window_content])
        
        # Handle end of sequence (pad with last frame)  
        elif end > T:
            padding_needed = end - T
            window_end = T
            # Pad by repeating the last frame
            pad_frames = np.tile(features[-1:], (padding_needed, 1))  # Repeat last frame
            window_content = features[start:window_end]
            window = np.vstack([window_content, pad_frames])
        
        # Normal case (no padding needed)
        else:
            window = features[start:end]
        
        # Transpose to match training format: (freq_bins, context_size)
        window = window.T
        X.append(window)
    
    return np.stack(X)

# Alternative version if you prefer to skip edge frames instead of padding:
def create_prediction_windows_safe(features, context_size=4):
    """
    Safe version that only creates windows for frames with full context
    (skips edges rather than padding)
    """
    X = []
    half = context_size // 2
    T = features.shape[0]
    
    # Only create windows for frames that have full context
    for t in range(half, T - half):
        window = features[t - half : t + half + 1]   # shape: (context_size, freq_bins)
        window = window.T  # transpose → (freq_bins, context_size)
        X.append(window)
    
    return np.stack(X)

def create_prediction_windows_copy(features, context_size=4):
    """
    Creates context windows by duplicating the center frame to fill the entire window.
    This is different from the sliding window approach.
    """
    X = []
    T = features.shape[0]
    
    for t in range(T):
        # Duplicate the current frame to fill the entire context window
        duplicated_frames = np.tile(features[t:t+1], (context_size + 1, 1))  # shape: (context_size, freq_bins)
        window = duplicated_frames.T  # transpose → (freq_bins, context_size)
        X.append(window)
    
    return np.stack(X)


if __name__ == "__main__":
    audio_path = "SampleAudioWav\Ditto arpeg.wav"  # Replace with your audio file path
    
    y, sr = dp.load_audio(audio_path)

    delta_t = 0.04

    # CQT params
    hop_length = int(sr * delta_t)
    n_bins = 115 # up to 20000 hz
    bins_per_octave = 12
    fmin = librosa.note_to_hz('A0')

    CQT_data, sr = dp.load_data(audio_path, hop_length=hop_length, n_bins=n_bins, 
                        bins_per_octave=bins_per_octave, fmin=fmin)
    print(f"CQT data shape: {CQT_data.shape}, Sample rate: {sr}")

    # Create windows for prediction
    prediction_windows = create_prediction_windows(CQT_data, context_size=4)
    print(f"Prediction windows shape: {prediction_windows.shape}")  # (time, freq_bins, context_size)

    # Add channel dimension for CNN input
    prediction_windows = prediction_windows[:, np.newaxis, :, :]  # (time, 1, freq_bins, context_size)
    #loaded = np.load("parsed data/normalization_params.npz", allow_pickle=True)
    #params = {key: loaded[key] for key in loaded.files}
    prediction_windows, _ = dp.normalize_data(prediction_windows)
    print(f"Final input shape: {prediction_windows.shape}")

    # nn model - input_shape should be (channels, freq_bins, context_size)
    model = PitchCNN(num_notes=88, input_shape=(prediction_windows.shape[1], prediction_windows.shape[2], prediction_windows.shape[3]))
    
    # Load model weights
    model.load_state_dict(torch.load('NN Saved Models\cnn_2convwithres_.9930\cnn_acc0.9930.pth'))
    model.eval()  # Set to evaluation mode
    
    # Calculate time for each frame (you need to know the hop_length used during training)
    # If you don't know, use the same as in your load_data function (delta_t=0.04)
    delta_t = 0.04  # This should match what you used in load_dataset
    hop_length = int(sr * delta_t)
    time_per_frame = hop_length / sr
    print(f"Hop length: {hop_length}, Time per frame: {time_per_frame:.4f}s")
    
    # Convert to tensor and make predictions
    input_tensor = torch.tensor(prediction_windows, dtype=torch.float32)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_notes = (predictions >= 0.5).int().numpy()
    
    print(f"Predictions shape: {predicted_notes.shape}")

    # Helper function to convert bin index to note name
    def bin_to_note_name(bin_idx):
        # Assuming bin 0 = A0 (MIDI 21), bin 1 = A#0, etc.
        midi_note = 21 + bin_idx  # A0 = MIDI 21
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note - 12) // 12
        note_name = note_names[midi_note % 12]
        return f"{note_name}{octave}"
    
    # Convert to time-based note events
    active_notes = {}
    for frame_idx in range(len(predicted_notes)):
        current_time = frame_idx * time_per_frame
        frame_prediction = predicted_notes[frame_idx]
        
        # Find which notes are active in this frame
        active_in_frame = np.where(frame_prediction == 1)[0]
        
        for note_bin in active_in_frame:
            note_name = bin_to_note_name(note_bin)
            if note_name not in active_notes:
                # Start a new note
                active_notes[note_name] = {'start': current_time, 'end': current_time, 'active': True}
            else:
                # Extend the existing note
                active_notes[note_name]['end'] = current_time
    
    # Print note events (filter out very short notes)
    min_note_duration = 0.00  # Minimum duration in seconds to consider a real note
    
    print("\nDetected Notes:")
    for note_name, times in active_notes.items():
        duration = times['end'] - times['start']
        if duration >= min_note_duration:
            print(f"{note_name}: {times['start']:.2f}s - {times['end']:.2f}s (duration: {duration:.2f}s)")
    
    # Also print some statistics
    total_notes = len([note for note in active_notes.values() if (note['end'] - note['start']) >= min_note_duration])
    print(f"\nTotal notes detected: {total_notes}")
    print(f"Total frames processed: {len(predicted_notes)}")
    
    # Optional: Print first few frames of predictions to see what's happening
    print("\nFirst 5 frames predictions (note indices):")
    for i in range(min(100, len(predicted_notes))):
        active_notes_in_frame = np.where(predicted_notes[i] == 1)[0]
        note_names = [bin_to_note_name(idx) for idx in active_notes_in_frame]
        print(f"Frame {i}: {note_names}")