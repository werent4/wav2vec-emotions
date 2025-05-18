# Speech Emotion Recognition with Wav2Vec2

A cross-platform solution for speech emotion recognition using Wav2Vec2 model with Python training pipeline and C++ inference.

## Overview

This project provides a complete pipeline for speech emotion recognition:

1. **Python components** for training and fine-tuning Wav2Vec2 models on emotion recognition tasks
2. **ONNX export** for cross-platform model compatibility
3. **C++ inference engine** for real-time emotion recognition applications

The system classifies spoken emotions into categories including: angry, sad, neutral, happy, excited, frustrated, fear, surprise, and disgust.

## Features

- Fine-tune Facebook's Wav2Vec2 models for emotion recognition
- Data augmentation techniques for improved generalization
- Focal Loss implementation for handling class imbalance
- Export models to ONNX format for deployment
- Fast C++ inference using ONNX Runtime
- Audio preprocessing and feature extraction
- Support for both CPU and GPU acceleration

## Project Structure

```
├── python/                 # Python training and export components
│   ├── configs/            # Configuration files
│   ├── src/                # Source code
│       ├── train.py        # Training script
│       ├── inference.py    # Python inference
│       ├── export.py       # Model export to ONNX
│       ├── dataset.py      # Dataset handling
│       ├── augments.py     # Audio augmentation
│       ├── loss_function.py # Focal loss implementation
│
├── cpp/                    # C++ inference engine
│   ├── include/            # Header files
│   ├── src/                # C++ source files
│   ├── CMakeLists.txt      # Build configuration
```

## Requirements

### Python
- Python 3.8+
- PyTorch
- Transformers
- Librosa
- NumPy
- scikit-learn
- datasets
- torchaudio

### C++
- CMake 3.10+
- ONNX Runtime
- libsndfile
- C++17 compatible compiler
- jsoncpp

## Installation

### Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install torch torchaudio transformers datasets librosa scikit-learn numpy
```

### C++ Build

```bash
cd cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
**NOTE:** Windows builds are note supported yet!

## Usage

### Training a Model

1. Prepare your configuration file (see `python/configs/train.json` for an example)
2. Run the training script:

```bash
python python/src/train.py --config python/configs/train.json
```

Optional parameters:
- `--model_name`: Specify a different base model (overrides config)
- `--output_dir`: Change the output directory (overrides config)

### Exporting to ONNX

After training, export your model to ONNX format:

```bash
python python/src/export.py --model_path output_model/best_f1 --output_path onnx_model
```

Parameters:
- `--model_path`: Path to the trained PyTorch model
- `--output_path`: Output directory for ONNX model
- `--sample_rate`: Audio sample rate (default: 16000)
- `--duration`: Maximum audio duration in seconds (default: 5)

### Inference with Python

```bash
python python/src/inference.py --model_name output_model/best_f1 --audio_file_path path/to/your/audio.wav
```

### Inference with C++

```bash
./build/wav2vec-emotions path/to/your/audio.wav onnx_model/metadata.json
```

## Training Configuration

The training configuration file (`train.json`) contains the following parameters:

```json
{
    "model_name": "facebook/wav2vec2-base",
    "seed": 42,
    "val_size": 0.15,
    "max_lenght_seconds": 5,
    "is_simplified": false,
    "gamma": 2.0,
    "alpha": 0.1, 
    "training_args": {
        "output_dir": "output_model",
        "save_strategy": "epoch",
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "lr_scheduler_type": "cosine", 
        "weight_decay": 0.04, 
        "warmup_ratio": 0.1,
        "logging_steps": 50,
        "eval_strategy": "epoch",
        "save_total_limit": 2,
        "fp16": true,
        "dataloader_num_workers": 8,
        "report_to": "none",
        "load_best_model_at_end": true,
        "metric_for_best_model": "f1"
    }
}
```

## Emotion Classes

The system classifies emotions into the following categories:

- angry (0)
- sad (1)
- neutral (2)
- happy (3)
- excited (4)
- frustrated (5)
- fear (6)
- surprise (7)
- disgust (8)
- unknown (9)

## Dataset

The system is designed to work with the IEMOCAP dataset, which can be loaded using the Hugging Face datasets library:

```python
dataset = load_dataset("AbstractTTS/IEMOCAP", trust_remote_code=True)
```

## Audio Augmentation

The system implements several audio augmentation techniques:

- Volume change
- Noise addition
- Tempo modification
- Pitch shifting

## Acknowledgements

- [Facebook Wav2Vec2](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)
- [IEMOCAP Dataset](https://sail.usc.edu/iemocap/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)

## TODO

- Add LibTorch backend support
- Add TensorRT backend support
- Add CUDA optimizations for preprocessing
- Add CUDA optimizations for postprocessing
- Add bacth support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
