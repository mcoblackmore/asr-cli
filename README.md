# asr-cli

CLI for local ASR on macOS, with:
- official `openai-whisper` support for Whisper models
- `transcribe` for audio/video transcription
- `rectify` for SRT correction using `opencli gemini ask`
- `all` for transcribe-then-rectify in one command
- `fcpxml` export for Final Cut Pro subtitle import

Default models:
- transcription: `openai/whisper-large-v3-turbo`
- exact subtitle alignment: `Qwen/Qwen3-ForcedAligner-0.6B`

Built-in shorthand aliases:
- `qwen` -> `mlx-community/Qwen3-ASR-0.6B-4bit`
- `glm` -> `mlx-community/GLM-ASR-Nano-2512-4bit`

## Runtime Dependencies

Core runtime dependencies from `pyproject.toml`:
- `openai-whisper>=20250625`
- `mlx-audio>=0.4.2`

System / environment dependencies:
- Python `>= 3.12`
- `ffmpeg` for common audio/video container decoding such as `.mp4`, `.mov`, `.mkv`
- `opencli` with a working Gemini web session for `rectify` and `all`

Backend usage in this repo:
- official OpenAI Whisper models use the official `openai-whisper` Python package
- Qwen3-ASR, GLM-ASR, and the forced aligner run through `mlx-audio`

## Referenced Core Projects

This project is built on top of, or directly references, these upstream GitHub projects:
- OpenAI Whisper: [openai/whisper](https://github.com/openai/whisper)
- MLX-Audio: [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- OpenCLI: [opencli-io/opencli](https://github.com/opencli-io/opencli)
- MLX: [ml-explore/mlx](https://github.com/ml-explore/mlx)

Model repos commonly used with this CLI:
- Whisper Turbo model repo: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- Qwen3-ASR MLX repo: [mlx-community/Qwen3-ASR-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit)
- GLM-ASR Nano MLX repo: [mlx-community/GLM-ASR-Nano-2512-4bit](https://huggingface.co/mlx-community/GLM-ASR-Nano-2512-4bit)
- Qwen3 Forced Aligner MLX repo: [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit)

## Quickstart

### Install

Run the macOS installer from the repo root:

```bash
./install-macos.sh
```

The installer will:
- check for Python `>= 3.12`
- stop and tell you how to install Python first if it is missing
- create the repo-local `.venv`
- install `asr-cli` into that `.venv`
- install a user launcher at `~/.local/bin/asr-cli`

If `~/.local/bin` is not already on your `PATH`, add it:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Full SRT Flow

To run a full transcribe + rectify flow and get SRT output:

```bash
asr-cli all input.mp4 --max-chars-per-line 24 --new-chat
```

This will:
1. transcribe `input.mp4` to an `.srt` next to the input file
2. send that SRT to Gemini through OpenCLI for correction
3. write a corrected `.correct.srt` next to the input file

Example output files:
- `input.srt`
- `input.correct.srt`

If those files already exist, `asr-cli` adds numeric suffixes instead of
overwriting them.

### Requirements For `rectify` / `all`

The correction step uses:

```bash
opencli gemini ask
```

So you need OpenCLI installed and a working Gemini web session in the browser
profile OpenCLI uses.

## Actions

### `transcribe`

Generate transcript output from an input media file.

Default transcription model:

```bash
openai/whisper-large-v3-turbo
```

Default aligner model used for exact SRT/FCPXML subtitle timing when needed:

```bash
Qwen/Qwen3-ForcedAligner-0.6B
```

```bash
asr-cli transcribe input.mp4 --format srt --max-chars-per-line 24
```

When `--max-chars-per-line` is enabled for `srt` or `fcpxml`, subtitle splitting
now follows two rules in order:
- never exceed the configured character limit
- otherwise prefer sentence-ending semantic boundaries from the backend, with
  punctuation-based fallback when sentence metadata is unavailable

Override the transcription model with `--model`:

```bash
asr-cli transcribe input.mp4 --model zai-org/GLM-ASR-Nano-2512 --format srt
```

Use Whisper Turbo:

```bash
asr-cli transcribe input.mp4 --model openai/whisper-large-v3-turbo --format srt
```

This uses the official `openai-whisper` Python package and the official
OpenAI Whisper Turbo model implementation.

By default, output is written next to the input file. If the target output file
already exists, a numeric suffix is added.

You can also export Final Cut Pro subtitles directly:

```bash
asr-cli transcribe input.mp4 --format fcpxml --max-chars-per-line 24
```

This writes an `.fcpxml` project containing caption elements with transparent
subtitle backgrounds so the file can be imported into Final Cut Pro.

### `rectify`

Read an existing `.srt`, send the full content to Gemini through OpenCLI, and
write a corrected `.correct.srt` while preserving the original cue indices and
timeline segments.

```bash
asr-cli rectify input.srt --wait-seconds 90 --new-chat
```

This action uses `opencli gemini ask`.

### `all`

Run the full flow in one command:
1. transcribe the input file with `openai/whisper-large-v3-turbo` to SRT
2. rectify that generated SRT with Gemini

```bash
asr-cli all input.mp4 --max-chars-per-line 24 --new-chat
```

Override the transcription model in the combined flow with `--model`:

```bash
asr-cli all input.mp4 --model zai-org/GLM-ASR-Nano-2512 --max-chars-per-line 24 --new-chat
```

Whisper Turbo also works in the combined flow:

```bash
asr-cli all input.mp4 --model openai/whisper-large-v3-turbo --max-chars-per-line 24 --new-chat
```
