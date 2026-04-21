# asr-cli (Windows Version)

CLI for local ASR on Windows, using Faster-Whisper for transcription.

## Features

- `transcribe` - Audio/video transcription to text, SRT, VTT, or JSON
- Multiple output formats: txt, json, srt, vtt
- SRT subtitle with configurable max characters per line
- Support for multiple Whisper model sizes: tiny, base, small, medium, large
- Chinese language support with auto-detection

## Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~75MB | Fastest | Basic |
| small | ~400MB | Medium | High |
| medium | ~1.4GB | Slow | Very High |
| large | ~2.9GB | Slowest | Highest |

## Requirements

- Python >= 3.12
- faster-whisper
- av (for audio/video decoding)

## Installation

### Option 1: Using install script (Recommended)

1. Clone or download this repository
2. Run the installer:
   ```cmd
   install-windows.bat
   ```

### Option 2: Manual Installation

```cmd
pip install faster-whisper av
pip install -e .
```

## Usage

### Basic transcription

```cmd
asr-cli transcribe audio.mp3 --format srt --model small
```

### With language specified

```cmd
asr-cli transcribe video.mp4 --format srt --language zh --model small
```

### With character limit per line

```cmd
asr-cli transcribe audio.mp3 --format srt --max-chars-per-line 24
```

### Different output formats

```cmd
asr-cli transcribe audio.mp3 --format txt
asr-cli transcribe audio.mp3 --format json
asr-cli transcribe audio.mp3 --format vtt
```

### Output to specific location

```cmd
asr-cli transcribe audio.mp3 --format srt --output C:\output\subtitles.srt
```

## Model Download

Models are downloaded automatically on first use. If you have a Hugging Face account with an access token, you can use it to speed up downloads:

```cmd
python -c "from huggingface_hub import login; login('your-token')"
```

Model cache location: `%USERPROFILE%\.cache\huggingface\hub`

## Model Sources

Models from Systran on Hugging Face:
- https://huggingface.co/Systran/faster-whisper-tiny
- https://huggingface.co/Systran/faster-whisper-small
- https://huggingface.co/Systran/faster-whisper-medium
- https://huggingface.co/Systran/faster-whisper-large

## Troubleshooting

### MKL malloc errors

If you encounter memory errors with small model, use tiny model instead:
```cmd
asr-cli transcribe audio.mp3 --format srt --model tiny
```

### Slow transcription

- Use `--model tiny` for faster results with lower accuracy
- Close other applications to free up RAM
- Consider using small model for better accuracy/speed balance

## License

MIT License (same as original OpenAI Whisper)