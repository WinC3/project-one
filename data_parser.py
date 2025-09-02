import librosa
import librosa.display
import numpy as np
import pandas as pd

import pretty_midi

import os

csv_metadata_path = "MAESTRO Data\maestro-v1.0.0\maestro-v1.0.0.csv"

def load_dataset(n_samples=None, shuffle=False, delta_t=0.02):
    metadata_df = pd.read_csv(csv_metadata_path)
    metadata_array = metadata_df.values

    if shuffle:
        np.random.shuffle(metadata_array)

    if n_samples is not None:
        metadata_array = metadata_array[:n_samples]

    all_data, all_labels = [], []
    
    for i, sample in enumerate(metadata_array):
        audio_path = os.path.normpath("MAESTRO Data/maestro-v1.0.0/" + sample[5])
        midi_path = os.path.normpath("MAESTRO Data/maestro-v1.0.0/" + sample[4])

        y, sr = load_audio(audio_path)
        if y is None or sr is None:
            continue

        # CQT params
        hop_length = int(sr * delta_t)
        n_bins = 88
        bins_per_octave = 12
        fmin = librosa.note_to_hz('A0')

        CQT_data, sr = load_data(audio_path, hop_length=hop_length, n_bins=n_bins, 
                           bins_per_octave=bins_per_octave, fmin=fmin)
        if CQT_data is None:
            continue
        
        # Load MIDI with note transition information
        labels, note_transitions = load_labels_with_transitions(midi_path, sr, hop_length, 
                                                              total_frames=CQT_data.shape[0], 
                                                              n_bins=n_bins, fmin=fmin)
        if labels is None:
            continue

        # REMOVE FRAMES WHERE NOTE TRANSITIONS OCCUR
        stable_mask = ~note_transitions  # True for stable frames, False for transition frames
        CQT_stable = CQT_data[stable_mask]
        labels_stable = labels[stable_mask]
        
        if len(CQT_stable) == 0:
            print(f"Skipping {audio_path} - no stable frames after transition removal")
            continue

        print(f"Original: {CQT_data.shape}, After transition removal: {CQT_stable.shape}")
        print(f"Removed {len(CQT_data) - len(CQT_stable)} transition frames")

        all_data.append(CQT_stable)
        all_labels.append(labels_stable)
        
        print(f"Added: {CQT_stable.shape} features, {labels_stable.shape} labels - File {i+1}")

    return all_data, all_labels

    #train_ratio = 0.7
    #val_ratio = 0.15

    # Split into train/validation/test
    #return split_data(all_data, all_labels, train_ratio, val_ratio)


def split_data_by_track(all_data, all_labels, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets by track"""
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


def split_data(all_data, all_labels, train_ratio=0.7, val_ratio=0.15, shuffle=True):
    """Split data into train, validation, and test sets by individual frames
    
    Parameters:
    all_data: list of numpy arrays or single numpy array where each row is a frame
    all_labels: list of numpy arrays or single numpy array where each row is a frame
    train_ratio: proportion of frames for training
    val_ratio: proportion of frames for validation
    
    Returns:
    Split arrays in the same format as input
    """
    
    # Handle both single array and list of arrays input
    if isinstance(all_data, np.ndarray):
        all_data = [all_data]
    if isinstance(all_labels, np.ndarray):
        all_labels = [all_labels]
    
    # Concatenate all frames into single arrays
    X_combined = np.concatenate(all_data, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)
    
    total_frames = len(X_combined)
    print(f"Total frames: {total_frames}")

    # Shuffle if requested
    if shuffle:
        print("Shuffling frames before splitting...")
        indices = np.random.permutation(total_frames)
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]
    
    # Calculate split indices
    train_end = int(total_frames * train_ratio)
    val_end = train_end + int(total_frames * val_ratio)
    
    # Split the frames
    train_data = X_combined[:train_end]
    train_labels = y_combined[:train_end]
    
    valid_data = X_combined[train_end:val_end]
    valid_labels = y_combined[train_end:val_end]
    
    test_data = X_combined[val_end:]
    test_labels = y_combined[val_end:]
    
    print(f"Split: {len(train_data)} train frames, {len(valid_data)} validation frames, {len(test_data)} test frames")
    
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


def load_labels_with_transitions(midi_path, sr, hop_length, total_frames, n_bins=88, fmin=27.5):
    """
    Load labels and also identify frames where note transitions occur
    """
    try:
        mid = pretty_midi.PrettyMIDI(midi_path)
    except:
        return None, None
    
    # Create label matrix and transition flag array
    labels = np.zeros((total_frames, n_bins), dtype=int)
    has_transition = np.zeros(total_frames, dtype=bool)  # True if frame has note change
    
    frame_duration = hop_length / sr  # Time per frame in seconds
    
    for instrument in mid.instruments:
        for note in instrument.notes:
            # Convert MIDI note to bin index
            bin_idx = note.pitch - 21  # A0 = MIDI 21 → bin 0
            if bin_idx < 0 or bin_idx >= n_bins:
                continue
            
            # Convert time to frame indices
            start_frame = int(note.start * sr / hop_length)
            end_frame = int(note.end * sr / hop_length)
            
            # Mark the frame where note STARTS as transition
            if start_frame < total_frames:
                has_transition[start_frame] = True
            
            # Mark the frame where note ENDS as transition  
            if end_frame < total_frames:
                has_transition[end_frame] = True
            
            # Mark active note duration (excluding transition frames)
            for frame in range(start_frame + 1, end_frame):  # Skip start frame
                if frame < total_frames:
                    labels[frame, bin_idx] = 1
    
    return labels, has_transition


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def save_parsed_data(n_samples=None):

    all_data, all_labels = load_dataset(n_samples=n_samples)

    X = np.concatenate(all_data, axis=0)    # Shape: (total_frames, 88)
    y = np.concatenate(all_labels, axis=0)  # Shape: (total_frames, 88)

    save_path = 'parsed data/'

    # save data
    #np.savez(save_path + 'data_by_entry.npz', **{f'entry{i}': arr for i, arr in enumerate(all_data)})
    #np.savez(save_path + 'label_by_entry.npz', **{f'entry{i}': arr for i, arr in enumerate(all_labels)})

    np.savez(save_path + 'cleaned_unseparated_dataset.npz', features=X, labels=y)


def load_dataset_from_file(load_path='parsed data/unseparated_dataset.npz', n_samples=None, shuffle=False):
    # Load with memory-mapping - doesn't load data until accessed
    data = np.load(load_path, mmap_mode='r')
    X = data['features'][:n_samples]
    y = data['labels'][:n_samples]
    print(f"Loaded data samples: {X.shape}")
    print(f"Loaded labels samples: {y.shape}")

    # Split into train/validation/test
    return split_data([X], [y], train_ratio=0.7, val_ratio=0.15, shuffle=shuffle)


def shuffle_file(load_path='parsed data/unseparated_dataset.npz', seed=42):
    data = np.load(load_path, mmap_mode='r')
    print("Loaded data for shuffling.")
    X = data['features']
    y = data['labels']
    total_frames = len(X)
    
    np.random.seed(seed)
    indices = np.random.permutation(total_frames)
    
    # Create memory-mapped output files
    X_shuffled = np.memmap('temp_X.dat', dtype=X.dtype, mode='w+', shape=X.shape)
    y_shuffled = np.memmap('temp_Y.dat', dtype=y.dtype, mode='w+', shape=y.shape)
    print("Created memory-mapped files for shuffled data.")
    
    # Shuffle in batches
    batch_size = 50000
    for i in range(0, total_frames, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_shuffled[i:i+batch_size] = X[batch_indices]
        y_shuffled[i:i+batch_size] = y[batch_indices]
    print("Shuffling complete.")
    
    # Save final result
    np.savez(f'parsed data/shuffled_dataset_seed_{seed}.npz', 
             features=X_shuffled, labels=y_shuffled)
    print(f"Shuffled dataset saved to 'parsed data/shuffled_dataset_seed_{seed}.npz'")
    
    # Cleanup temp files
    del X_shuffled, y_shuffled
    os.remove('temp_X.dat')
    os.remove('temp_Y.dat')
    print("Temporary files removed.")


if __name__ == "__main__":
    #shuffle_file(seed=0)
    save_parsed_data()
    #load_dataset_from_file()