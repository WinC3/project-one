import torch
import torchaudio
import numpy as np
import pretty_midi
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

from .model.nn_models import OnsetsAndFrames
from .audio.data_preprocessing import compute_log_mel

class PianoTranscriber:
    """
    Inference engine for piano transcription using trained Onsets and Frames model.
    
    Provides a clean interface for loading models and transcribing audio files,
    designed to be used by both CLI and GUI applications.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the transcriber.
        
        Args:
            checkpoint_path: Path to model checkpoint. If None, must call load_model() later.
            device: Device to run inference on. If None, auto-detects CUDA availability.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        
        # Audio processing constants (must match training config)
        self.sample_rate = 16000
        self.hop_length = 512
        self.sequence_length = 640  # ~20 seconds
        
        if checkpoint_path:
            self.load_model(checkpoint_path)
    
    def load_model(self, checkpoint_path: str) -> None:
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            print(f"Loading model from {checkpoint_path}")
            
            # Initialize model
            self.model = OnsetsAndFrames().to(self.device)
            
            # Load checkpoint (inference-only, no optimizer state needed)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_loaded = True
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _preprocess_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess audio file using same pipeline as training.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio features (mel spectrogram)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Reuse preprocessing from training pipeline
        return compute_log_mel(str(audio_path))
    
    def _chunk_audio(self, audio_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Split long audio into manageable chunks for inference.
        
        Args:
            audio_features: Full audio mel spectrogram (T, F)
            
        Returns:
            List of audio chunks, each of length sequence_length
        """
        total_frames = audio_features.shape[0]
        
        if total_frames <= self.sequence_length:
            # Pad if shorter than sequence length
            pad_len = self.sequence_length - total_frames
            padded = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
            return [padded]
        
        # Split into overlapping chunks to avoid boundary effects
        chunks = []
        stride = self.sequence_length // 2  # 50% overlap
        
        for start in range(0, total_frames - self.sequence_length + 1, stride):
            end = start + self.sequence_length
            chunk = audio_features[start:end]
            chunks.append(chunk)
        
        return chunks
    
    def _merge_predictions(self, chunk_predictions: List[Dict[str, torch.Tensor]], 
                          original_length: int) -> Dict[str, torch.Tensor]:
        """
        Merge overlapping chunk predictions back into full-length predictions.
        
        Args:
            chunk_predictions: List of prediction dictionaries from each chunk
            original_length: Original audio length in frames
            
        Returns:
            Merged predictions dictionary
        """
        if len(chunk_predictions) == 1:
            # Single chunk, just trim to original length
            pred = chunk_predictions[0]
            return {
                'onset': pred['onset'][:original_length],
                'frame': pred['frame'][:original_length], 
                'velocity': pred['velocity'][:original_length]
            }
        
        # Initialize output arrays
        onset_pred = torch.zeros(original_length, 88)
        frame_pred = torch.zeros(original_length, 88)
        velocity_pred = torch.zeros(original_length, 88)
        counts = torch.zeros(original_length, 88)
        
        stride = self.sequence_length // 2
        
        # Accumulate predictions with overlap handling
        for i, pred in enumerate(chunk_predictions):
            start = i * stride
            end = min(start + self.sequence_length, original_length)
            chunk_length = end - start
            
            onset_pred[start:end] += pred['onset'][:chunk_length]
            frame_pred[start:end] += pred['frame'][:chunk_length]
            velocity_pred[start:end] += pred['velocity'][:chunk_length]
            counts[start:end] += 1
        
        # Average overlapping regions
        onset_pred = onset_pred / (counts + 1e-8)
        frame_pred = frame_pred / (counts + 1e-8)
        velocity_pred = velocity_pred / (counts + 1e-8)
        
        return {
            'onset': onset_pred,
            'frame': frame_pred,
            'velocity': velocity_pred
        }
    
    def transcribe_audio(self, audio_path: Union[str, Path], 
                        onset_threshold: float = 0.5,
                        frame_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            onset_threshold: Threshold for onset detection
            frame_threshold: Threshold for frame detection
            
        Returns:
            Dictionary with 'onset', 'frame', 'velocity' predictions (T, 88)
            
        Raises:
            RuntimeError: If model not loaded
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess audio
        audio_features = self._preprocess_audio(audio_path)
        original_length = audio_features.shape[0]
        
        # Split into chunks if needed
        chunks = self._chunk_audio(audio_features)
        
        # Run inference on each chunk
        chunk_predictions = []
        
        with torch.no_grad():
            for chunk in chunks:
                # Add batch dimension and move to device
                chunk_batch = chunk.unsqueeze(0).to(self.device)
                
                # Model inference
                outputs = self.model(chunk_batch)
                
                # Apply sigmoid and move back to CPU
                predictions = {
                    'onset': torch.sigmoid(outputs['onset']).cpu().squeeze(0),
                    'frame': torch.sigmoid(outputs['frame']).cpu().squeeze(0),
                    'velocity': torch.sigmoid(outputs['velocity']).cpu().squeeze(0)
                }
                
                chunk_predictions.append(predictions)
        
        # Merge chunks back together
        merged_predictions = self._merge_predictions(chunk_predictions, original_length)
        
        # Apply thresholds
        merged_predictions['onset'] = (merged_predictions['onset'] > onset_threshold).float()
        merged_predictions['frame'] = (merged_predictions['frame'] > frame_threshold).float()
        
        return merged_predictions
    
    def predictions_to_midi(self, predictions: Dict[str, torch.Tensor], 
                           output_path: Optional[Union[str, Path]] = None) -> pretty_midi.PrettyMIDI:
        """
        Convert model predictions to MIDI format.
        
        Args:
            predictions: Predictions from transcribe_audio()
            output_path: Optional path to save MIDI file
            
        Returns:
            PrettyMIDI object
        """
        onset_pred = predictions['onset']
        frame_pred = predictions['frame'] 
        velocity_pred = predictions['velocity']
        
        # Create MIDI object
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        # Convert frame indices to time
        frame_time = self.hop_length / self.sample_rate
        
        # Extract notes from predictions
        for pitch_idx in range(88):
            pitch = pitch_idx + 21  # MIDI note number (A0 = 21)
            
            # Find onsets for this pitch
            onset_frames = torch.where(onset_pred[:, pitch_idx] == 1)[0]
            
            for onset_frame in onset_frames:
                onset_time = onset_frame.item() * frame_time
                velocity = max(1, min(127, int(velocity_pred[onset_frame, pitch_idx].item() * 127)))
                
                # Find note end (when frame prediction stops)
                end_frame = onset_frame
                for f in range(onset_frame, len(frame_pred)):
                    if frame_pred[f, pitch_idx] == 1:
                        end_frame = f
                    else:
                        break
                
                end_time = end_frame.item() * frame_time
                
                # Only add notes with reasonable duration
                if end_time > onset_time:
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=onset_time,
                        end=end_time
                    )
                    piano.notes.append(note)
        
        midi.instruments.append(piano)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            midi.write(str(output_path))
            print(f"MIDI saved to {output_path}")
        
        return midi
    
    def predictions_to_json(self, predictions: Dict[str, torch.Tensor]) -> List[Dict]:
        """
        Convert predictions to JSON-serializable format.
        
        Args:
            predictions: Predictions from transcribe_audio()
            
        Returns:
            List of note dictionaries with timing and velocity info
        """
        onset_pred = predictions['onset']
        frame_pred = predictions['frame']
        velocity_pred = predictions['velocity']
        
        frame_time = self.hop_length / self.sample_rate
        notes = []
        
        for pitch_idx in range(88):
            pitch = pitch_idx + 21
            onset_frames = torch.where(onset_pred[:, pitch_idx] == 1)[0]
            
            for onset_frame in onset_frames:
                onset_time = onset_frame.item() * frame_time
                velocity = velocity_pred[onset_frame, pitch_idx].item()
                
                # Find note end
                end_frame = onset_frame
                for f in range(onset_frame, len(frame_pred)):
                    if frame_pred[f, pitch_idx] == 1:
                        end_frame = f
                    else:
                        break
                
                end_time = end_frame.item() * frame_time
                
                if end_time > onset_time:
                    note = {
                        'pitch': pitch,
                        'start_time': onset_time,
                        'end_time': end_time,
                        'velocity': velocity,
                        'duration': end_time - onset_time
                    }
                    notes.append(note)
        
        return sorted(notes, key=lambda x: x['start_time'])


def get_latest_checkpoint(checkpoint_dir: Union[str, Path] = "transcriber/model") -> Optional[str]:
    """
    Find the latest model checkpoint in the model directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("model_epoch_*.pth"))
    
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    latest_epoch = 0
    latest_file = None
    
    for file in checkpoint_files:
        try:
            # Extract epoch number from filename
            epoch_str = file.stem.split('_')[-1]
            epoch = int(epoch_str)
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = file
        except (ValueError, IndexError):
            continue
    
    return str(latest_file) if latest_file else None


# Convenience function for quick transcription
def transcribe_file(audio_path: Union[str, Path], 
                   checkpoint_path: Optional[str] = None,
                   output_format: str = "midi",
                   output_path: Optional[Union[str, Path]] = None,
                   **kwargs) -> Union[pretty_midi.PrettyMIDI, List[Dict]]:
    """
    High-level function to transcribe a single audio file.
    
    Args:
        audio_path: Path to input audio file
        checkpoint_path: Path to model checkpoint (auto-detects if None)
        output_format: Output format ("midi" or "json")
        output_path: Output file path (optional)
        **kwargs: Additional arguments for transcribe_audio()
        
    Returns:
        MIDI object or list of note dictionaries
    """
    # Auto-detect checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint found. Please specify checkpoint_path.")
    
    # Initialize transcriber and run inference
    transcriber = PianoTranscriber(checkpoint_path)
    predictions = transcriber.transcribe_audio(audio_path, **kwargs)
    
    # Convert to requested format
    if output_format.lower() == "midi":
        return transcriber.predictions_to_midi(predictions, output_path)
    elif output_format.lower() == "json":
        result = transcriber.predictions_to_json(predictions)
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"JSON saved to {output_path}")
        return result
    else:
        raise ValueError(f"Unsupported output format: {output_format}")