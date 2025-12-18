# Piano Transcriber

A PyTorch implementation of the "Onsets and Frames" model for automatic piano transcription. Convert piano audio recordings into MIDI files using a combination of convolutional, recurrent, and traditional neural networks.

## Features

- **Transcription**: Uses the "Onsets and Frames" neural network architecture
- **Multiple output formats**: Generate MIDI files or JSON data with note timings
- **Command-line interface**: Easy-to-use CLI for batch processing
- **GPU acceleration**: Supports CUDA for faster inference
- **Flexible thresholds**: Adjustable onset and frame detection sensitivity

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/winc3/project-one.git
```

### For Development

```bash
git clone https://github.com/winc3/project-one.git
cd project-one
pip install -e .
```

## Usage

### Command Line Interface

**Basic usage:**
```bash
# Transcribe to MIDI
piano-transcriber input.wav -o output.mid

# Transcribe to JSON
piano-transcriber input.wav -f json -o output.json
```

**Batch processing:**
```bash
# Process multiple files
piano-transcriber *.wav -o /path/to/output/

# Process with custom sensitivity
piano-transcriber input.wav --onset-threshold 0.3 --frame-threshold 0.4
```

**Advanced options:**
```bash
# Use specific model
piano-transcriber input.wav -m path/to/model.pth

# Force CPU usage
piano-transcriber input.wav --device cpu

# Verbose output
piano-transcriber input.wav -v
```

### Python API

```python
from piano_transcriber import PianoTranscriber

# Initialize transcriber
transcriber = PianoTranscriber("path/to/model.pth")

# Transcribe audio file
predictions = transcriber.transcribe_audio("input.wav")

# Convert to MIDI
midi = transcriber.predictions_to_midi(predictions, "output.mid")

# Convert to JSON
notes = transcriber.predictions_to_json(predictions)
```

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3) 
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)

## Model Requirements

The transcriber requires a trained model checkpoint. You can:

1. **Train your own model** using the research components in this repository with your own data
2. **Use a pre-trained model** a sublist of model checkpoints from training included in `piano_transcriber/model/sample_checkpoints/` - ⚠️ **Non-commercial use only**
3. **Use the included model** in `piano_transcriber/model/` (if present) - ⚠️ **Non-commercial use only**

**⚠️ Important**: Pre-trained models included with this package were trained on the MAESTRO dataset and are restricted to non-commercial use. For commercial applications, you must train your own models using appropriately licensed data.

## Technical Details

- **Architecture**: Onsets and Frames neural network with CNN feature extraction and bidirectional LSTM
- **Input**: 16kHz audio, 229 mel-frequency bins
- **Output**: 88 piano keys (A0-C8, MIDI notes 21-108)
- **Inference**: Supports variable-length audio with chunking and overlap handling

## Contributing

Please feel free to submit issues, feature requests, or pull requests.

## License

**Code**: MIT License - see LICENSE file for details. The source code can be used for any purpose, including commercial applications.

**Pre-trained Models**: Models trained on the MAESTRO dataset are restricted to **non-commercial use only** due to dataset licensing terms. For commercial applications, you must train your own models using commercially-licensed data.

## Citation

This implementation is based on the "Onsets and Frames" model for automatic music transcription. If you use this code in your research, please consider citing the original paper from the original authors (of which I am not a part of).

**Reference**: [Onsets and Frames: Dual-Objective Piano Transcription](https://magenta.withgoogle.com/onsets-frames)

## Acknowledgments

- Built with PyTorch and torchaudio
- Uses the MAESTRO dataset for training
- Inspired by Google's Onsets and Frames implementation