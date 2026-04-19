from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import re
import subprocess
import sys
import tempfile
import warnings
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any


MODEL_ALIASES = {
    "glm": "mlx-community/GLM-ASR-Nano-2512-4bit",
    "zai-org/glm-asr-nano-2512": "mlx-community/GLM-ASR-Nano-2512-4bit",
    "zai-org/GLM-ASR-Nano-2512": "mlx-community/GLM-ASR-Nano-2512-4bit",
    "aligner": "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
    "qwen/qwen3-forcedaligner-0.6b": "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
    "Qwen/Qwen3-ForcedAligner-0.6B": "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
    "qwen": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "qwen/qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "Qwen/Qwen3-ASR-0.6B": "mlx-community/Qwen3-ASR-0.6B-4bit",
}

FORMAT_CHOICES = ("txt", "json", "srt", "vtt", "fcpxml")
FFMPEG_INPUT_SUFFIXES = {
    ".aac",
    ".aiff",
    ".avi",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".ogg",
    ".opus",
    ".webm",
    ".wma",
}
OFFICIAL_WHISPER_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "turbo",
    "large-v1",
    "large-v2",
    "large-v3",
}
OFFICIAL_WHISPER_REPO_TO_MODEL = {
    "openai/whisper-large-v3-turbo": "turbo",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v2": "large-v2",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr-cli",
        description=(
            "ASR CLI with official OpenAI Whisper support and MLX backends "
            "for Qwen3-ASR, GLM-ASR, and related checkpoints."
        ),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio or video")
    transcribe_parser.add_argument(
        "audio", nargs="+", help="Audio or video file(s) to transcribe"
    )
    transcribe_parser.add_argument(
        "--model",
        default="openai/whisper-large-v3-turbo",
        help=(
            "Official OpenAI Whisper model repo/name, MLX-community repo, or local "
            "path. Examples: openai/whisper-large-v3-turbo, turbo, "
            "Qwen/Qwen3-ASR-0.6B, zai-org/GLM-ASR-Nano-2512."
        ),
    )
    transcribe_parser.add_argument(
        "--resolved-model",
        action="store_true",
        help="Print the resolved model id that will actually be loaded and exit",
    )
    transcribe_parser.add_argument(
        "--list-models",
        action="store_true",
        help="List built-in model aliases and exit",
    )
    transcribe_parser.add_argument(
        "--output",
        "-o",
        default=".",
        help=(
            "Output directory or filename prefix. "
            "If this is a directory or ends with '/', each input uses its stem."
        ),
    )
    transcribe_parser.add_argument(
        "--format",
        "-f",
        default="txt",
        choices=FORMAT_CHOICES,
        help="Output format",
    )
    transcribe_parser.add_argument(
        "--language", default=None, help="Language hint, e.g. en or zh"
    )
    transcribe_parser.add_argument(
        "--context",
        default=None,
        help="Optional hotwords or context string passed to the backend when supported",
    )
    transcribe_parser.add_argument(
        "--prompt",
        default=None,
        help="Optional backend prompt or instruction string when supported",
    )
    transcribe_parser.add_argument(
        "--spelling-prompt",
        default=None,
        help="Optional spelling guidance for easily-mistaken words or names",
    )
    transcribe_parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Chunk duration in seconds when supported by the selected model",
    )
    transcribe_parser.add_argument(
        "--frame-threshold",
        type=int,
        default=None,
        help="Frame threshold for timestamp/segmentation heuristics when supported",
    )
    transcribe_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens when supported by the selected model",
    )
    transcribe_parser.add_argument(
        "--max-chars-per-line",
        type=int,
        default=None,
        help="Maximum characters per line in SRT or FCPXML subtitle output",
    )
    transcribe_parser.add_argument(
        "--aligner-model",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help=(
            "Forced aligner model used to obtain exact word timings for SRT splitting "
            "when the transcription model does not return word-level timestamps."
        ),
    )
    transcribe_parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming generation when supported by the selected backend",
    )
    transcribe_parser.add_argument(
        "--gen-kwargs",
        default=None,
        help='Extra backend kwargs as JSON, e.g. \'{"min_chunk_duration": 1.0}\'',
    )
    transcribe_parser.add_argument(
        "--verbose", action="store_true", help="Verbose backend output"
    )

    rectify_parser = subparsers.add_parser(
        "rectify",
        help="Correct wrong subtitle words with Gemini while preserving cue timing",
    )
    rectify_parser.add_argument("srt", nargs="+", help="SRT file(s) to correct")
    rectify_parser.add_argument(
        "--wait-seconds",
        type=int,
        default=90,
        help="Maximum seconds to wait for Gemini to return corrected SRT",
    )
    rectify_parser.add_argument(
        "--new-chat",
        action="store_true",
        help="Start a new Gemini chat before sending the prompt",
    )
    rectify_parser.add_argument(
        "--verbose", action="store_true", help="Verbose OpenCLI output"
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Transcribe with Whisper Turbo to SRT, then rectify it with Gemini",
    )
    all_parser.add_argument("input", help="Input audio or video file")
    all_parser.add_argument(
        "--model",
        default="openai/whisper-large-v3-turbo",
        help=(
            "Official OpenAI Whisper model repo/name, MLX-community repo, or local "
            "path. Defaults to OpenAI Whisper Turbo for the combined flow."
        ),
    )
    all_parser.add_argument(
        "--output",
        "-o",
        default=".",
        help=(
            "Output directory or filename prefix for the intermediate SRT. "
            "If this is a directory or ends with '/', the input stem is used."
        ),
    )
    all_parser.add_argument(
        "--max-chars-per-line",
        type=int,
        default=None,
        help="Maximum characters per line in SRT subtitle output",
    )
    all_parser.add_argument(
        "--aligner-model",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help=(
            "Forced aligner model used to obtain exact word timings for SRT splitting "
            "when the transcription model does not return word-level timestamps."
        ),
    )
    all_parser.add_argument(
        "--language", default=None, help="Language hint, e.g. en or zh"
    )
    all_parser.add_argument(
        "--context",
        default=None,
        help="Optional hotwords or context string passed to the transcription backend",
    )
    all_parser.add_argument(
        "--prompt",
        default=None,
        help="Optional backend prompt or instruction string when supported",
    )
    all_parser.add_argument(
        "--spelling-prompt",
        default=None,
        help="Optional spelling guidance for easily-mistaken words or names",
    )
    all_parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Chunk duration in seconds when supported by the selected model",
    )
    all_parser.add_argument(
        "--frame-threshold",
        type=int,
        default=None,
        help="Frame threshold for timestamp/segmentation heuristics when supported",
    )
    all_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens when supported by the selected model",
    )
    all_parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming generation when supported by the selected model",
    )
    all_parser.add_argument(
        "--gen-kwargs",
        default=None,
        help='Extra backend kwargs as JSON, e.g. \'{"min_chunk_duration": 1.0}\'',
    )
    all_parser.add_argument(
        "--wait-seconds",
        type=int,
        default=90,
        help="Maximum seconds to wait for Gemini to return corrected SRT",
    )
    all_parser.add_argument(
        "--new-chat",
        action="store_true",
        help="Start a new Gemini chat before sending the prompt",
    )
    all_parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )
    return parser


def normalize_model(model: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path)
    return MODEL_ALIASES.get(model, MODEL_ALIASES.get(model.lower(), model))


def parse_gen_kwargs(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --gen-kwargs JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--gen-kwargs must decode to a JSON object")
    return parsed


def print_model_aliases() -> None:
    print("Built-in model aliases:")
    print("  qwen -> mlx-community/Qwen3-ASR-0.6B-4bit")
    print("  Qwen/Qwen3-ASR-0.6B -> mlx-community/Qwen3-ASR-0.6B-4bit")
    print("  glm -> mlx-community/GLM-ASR-Nano-2512-4bit")
    print("  zai-org/GLM-ASR-Nano-2512 -> mlx-community/GLM-ASR-Nano-2512-4bit")
    print("  Official OpenAI Whisper default: openai/whisper-large-v3-turbo")


def resolve_output_prefix(audio_path: Path, output_arg: str, multiple_inputs: bool) -> str:
    output_path = Path(output_arg).expanduser()
    if output_arg in {".", "./"}:
        return str(audio_path.parent / audio_path.stem)

    treat_as_dir = (
        output_arg.endswith(("/", "\\"))
        or output_path.exists() and output_path.is_dir()
        or multiple_inputs
    )
    if treat_as_dir:
        return str(output_path / audio_path.stem)
    if output_path.suffix:
        return str(output_path.with_suffix(""))
    return str(output_path)


def uniquify_output_prefix(output_prefix: str, output_format: str) -> str:
    output_path = Path(output_prefix)
    candidate = output_path
    counter = 2
    while candidate.with_suffix(f".{output_format}").exists():
        candidate = output_path.with_name(f"{output_path.name}-{counter}")
        counter += 1
    return str(candidate)


def uniquify_path(path: Path) -> Path:
    candidate = path
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        counter += 1
    return candidate


def confirm_overwrite(path: Path) -> bool:
    while True:
        answer = input(f"Output file exists: {path}\nOverwrite? [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please answer y or n.", file=sys.stderr)


def choose_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    if confirm_overwrite(path):
        return path
    return uniquify_path(path)


def choose_output_prefix(output_prefix: str, output_format: str) -> str:
    output_path = choose_output_path(Path(output_prefix).with_suffix(f".{output_format}"))
    return str(output_path.with_suffix(""))


def print_run_config(title: str, items: list[tuple[str, Any]]) -> None:
    print(f"=== {title} ===", file=sys.stderr)
    for key, value in items:
        print(f"{key}: {value}", file=sys.stderr)
    print(file=sys.stderr)


def join_prompt_parts(*parts: str | None) -> str | None:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return None
    return "\n\n".join(cleaned)


def build_spelling_guidance(spelling_prompt: str | None) -> str | None:
    if spelling_prompt is None or not spelling_prompt.strip():
        return None
    return (
        "Use these spelling hints when transcribing names, brands, proper nouns, "
        "and other easily mistaken words:\n"
        f"{spelling_prompt.strip()}"
    )


def build_official_whisper_prompt(args: argparse.Namespace) -> str | None:
    return join_prompt_parts(
        args.context,
        args.prompt,
        build_spelling_guidance(getattr(args, "spelling_prompt", None)),
    )


def build_mlx_prompt_kwargs(
    resolved_model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    lowered = resolved_model.lower()
    spelling_guidance = build_spelling_guidance(getattr(args, "spelling_prompt", None))
    combined_instruction = join_prompt_parts(args.context, args.prompt, spelling_guidance)

    if "qwen3-asr" in lowered:
        return {"system_prompt": combined_instruction} if combined_instruction else {}
    if "glm-asr" in lowered:
        return {"prompt": combined_instruction} if combined_instruction else {}

    kwargs: dict[str, Any] = {}
    if args.context is not None:
        kwargs["context"] = args.context
    if args.prompt is not None:
        kwargs["prompt"] = args.prompt
    if spelling_guidance is not None:
        kwargs["prompt"] = join_prompt_parts(kwargs.get("prompt"), spelling_guidance)
    return kwargs


def load_backend():
    from mlx_audio.stt.generate import generate_transcription
    from mlx_audio.stt.utils import load_model

    return load_model, generate_transcription


def is_official_whisper_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return lowered in OFFICIAL_WHISPER_MODELS or lowered in OFFICIAL_WHISPER_REPO_TO_MODEL


def official_whisper_runtime_model(model_name: str) -> str:
    lowered = model_name.lower()
    return OFFICIAL_WHISPER_REPO_TO_MODEL.get(lowered, lowered)


def load_official_whisper() -> Any:
    try:
        import whisper
    except ImportError as exc:
        raise SystemExit(
            "Official Whisper support requires the openai-whisper package. "
            "Install it in this environment first."
        ) from exc
    return whisper


def transcribe_with_official_whisper(
    whisper_module: Any,
    model: Any,
    audio_path: str,
    args: argparse.Namespace,
) -> SimpleNamespace:
    if args.stream:
        raise SystemExit("--stream is not supported by the official Whisper backend.")
    if args.frame_threshold is not None:
        raise SystemExit("--frame-threshold is not supported by the official Whisper backend.")
    if args.max_tokens is not None:
        raise SystemExit("--max-tokens is not supported by the official Whisper backend.")
    if args.gen_kwargs is not None:
        raise SystemExit("--gen-kwargs is not supported by the official Whisper backend.")
    if args.chunk_duration is not None:
        raise SystemExit("--chunk-duration is not supported by the official Whisper backend.")

    transcribe_kwargs: dict[str, Any] = {
        "verbose": args.verbose,
        "word_timestamps": args.format in {"srt", "vtt", "fcpxml"}
        or args.max_chars_per_line is not None,
    }
    if args.language is not None:
        transcribe_kwargs["language"] = args.language
    initial_prompt = build_official_whisper_prompt(args)
    if initial_prompt is not None:
        transcribe_kwargs["initial_prompt"] = initial_prompt
    raw_result = model.transcribe(audio_path, **transcribe_kwargs)
    return SimpleNamespace(
        text=str(raw_result.get("text", "")).strip(),
        segments=raw_result.get("segments") or [],
        language=raw_result.get("language"),
        raw=raw_result,
    )


def infer_whisper_processor_repo(model_name: str) -> str | None:
    lowered = model_name.lower()
    if "whisper-large-v3-turbo" in lowered:
        return "openai/whisper-large-v3-turbo"
    if "whisper-large-v3" in lowered:
        return "openai/whisper-large-v3"
    if "whisper-large-v2" in lowered:
        return "openai/whisper-large-v2"
    if "distil-large-v3" in lowered:
        return "distil-whisper/distil-large-v3"
    return None


def ensure_whisper_processor(model: Any, model_name: str) -> None:
    if "whisper" not in model_name.lower():
        return
    if getattr(model, "_processor", None) is not None:
        return

    processor_repo = infer_whisper_processor_repo(model_name)
    if processor_repo is None:
        return

    try:
        from transformers import WhisperProcessor

        model._processor = WhisperProcessor.from_pretrained(processor_repo)
    except Exception as exc:
        raise SystemExit(
            "This Whisper model is missing local processor files, and asr-cli could not "
            f"load a compatible processor from {processor_repo}: {exc}"
        ) from exc


def load_transcription_model(load_model: Any, model_name: str) -> Any:
    if "whisper" not in model_name.lower():
        return load_model(model_name)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Could not load WhisperProcessor: .*",
            category=UserWarning,
        )
        model = load_model(model_name)
    ensure_whisper_processor(model, model_name)
    return model


SRT_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


def format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def parse_srt_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, milliseconds = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


def format_vtt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes:02d}:{seconds:06.3f}"


SENTENCE_BREAK_CHARS = {"。", "！", "？", "!", "?", "."}
ELLIPSIS_BREAK_TAILS = ("...", "…", "……")
ASCII_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9]")


def visible_length(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def combine_cue_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if ASCII_WORD_CHAR_RE.fullmatch(left[-1]) and ASCII_WORD_CHAR_RE.fullmatch(right[0]):
        return f"{left} {right}"
    return f"{left}{right}"


def text_has_sentence_break(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(ELLIPSIS_BREAK_TAILS):
        return True
    return stripped[-1] in SENTENCE_BREAK_CHARS


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_srt_entries(content: str) -> list[dict[str, str]]:
    normalized = content.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    entries: list[dict[str, str]] = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise SystemExit("Invalid SRT: each cue must contain index, timeline, and text.")
        match = SRT_TIMESTAMP_RE.fullmatch(lines[1].strip())
        if match is None:
            raise SystemExit(f"Invalid SRT timeline: {lines[1]}")
        entries.append(
            {
                "index": lines[0].strip(),
                "start": match.group("start"),
                "end": match.group("end"),
                "text": "\n".join(lines[2:]).strip(),
            }
        )
    return entries


def render_srt_entries(entries: list[dict[str, str]]) -> str:
    blocks = []
    for entry in entries:
        blocks.append(
            "\n".join(
                [
                    entry["index"],
                    f"{entry['start']} --> {entry['end']}",
                    entry["text"],
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def render_vtt_entries(entries: list[dict[str, str]]) -> str:
    blocks = ["WEBVTT"]
    for entry in entries:
        start = format_vtt_timestamp(parse_srt_timestamp(entry["start"]))
        end = format_vtt_timestamp(parse_srt_timestamp(entry["end"]))
        blocks.append(f"{start} --> {end}\n{entry['text']}")
    return "\n\n".join(blocks) + "\n"


def normalize_caption_language(language: str | None) -> str:
    if language is None or not language.strip():
        return "und"
    cleaned = language.strip().replace("_", "-")
    if "-" in cleaned:
        return cleaned
    simple_map = {
        "en": "en-US",
        "zh": "zh-CN",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "fr": "fr-FR",
        "de": "de-DE",
        "es": "es-ES",
    }
    return simple_map.get(cleaned.lower(), cleaned)


def format_fcpxml_time(seconds: float) -> str:
    milliseconds = max(int(round(seconds * 1000)), 0)
    if milliseconds == 0:
        return "0s"
    denominator = 1000
    gcd = math.gcd(milliseconds, denominator)
    numerator = milliseconds // gcd
    denominator //= gcd
    if denominator == 1:
        return f"{numerator}s"
    return f"{numerator}/{denominator}s"


def render_fcpxml_subtitles(
    entries: list[dict[str, str]],
    project_name: str,
    language: str | None,
) -> str:
    total_duration = 0.0
    for entry in entries:
        end_seconds = parse_srt_timestamp(entry["end"])
        total_duration = max(total_duration, end_seconds)

    role_language = normalize_caption_language(language)
    captions: list[str] = []
    for index, entry in enumerate(entries, start=1):
        start_seconds = parse_srt_timestamp(entry["start"])
        end_seconds = parse_srt_timestamp(entry["end"])
        duration_seconds = max(end_seconds - start_seconds, 0.001)
        text = html.escape(entry["text"])
        captions.append(
            "          "
            f'<caption name="Subtitle {index}" lane="1" offset="{format_fcpxml_time(start_seconds)}" '
            f'start="0s" duration="{format_fcpxml_time(duration_seconds)}" '
            f'role="asr-cli.subtitle?captionFormat=ITT.{role_language}">\n'
            '            <text placement="bottom">\n'
            '              <text-style fontFace="Regular" fontColor="1 1 1 1" '
            f'backgroundColor="0 0 0 0">{text}</text-style>\n'
            "            </text>\n"
            "          </caption>"
        )

    project_name_xml = html.escape(project_name)
    total_duration_xml = format_fcpxml_time(total_duration if total_duration > 0 else 0.001)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE fcpxml>\n"
        '<fcpxml version="1.10">\n'
        "  <resources>\n"
        '    <format id="r1" name="FFVideoFormat1080p30" frameDuration="100/3000s" '
        'width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>\n'
        "  </resources>\n"
        "  <library>\n"
        '    <event name="asr-cli">\n'
        f'      <project name="{project_name_xml}">\n'
        f'        <sequence format="r1" duration="{total_duration_xml}" tcStart="0s" '
        'tcFormat="NDF" audioLayout="stereo" audioRate="48k">\n'
        "          <spine>\n"
        f'            <gap name="Subtitles" offset="0s" start="0s" duration="{total_duration_xml}">\n'
        f'{"\n".join(captions)}\n'
        "            </gap>\n"
        "          </spine>\n"
        "        </sequence>\n"
        "      </project>\n"
        "    </event>\n"
        "  </library>\n"
        "</fcpxml>\n"
    )


def corrected_srt_path(input_path: Path) -> Path:
    return choose_output_path(input_path.with_name(f"{input_path.stem}.correct{input_path.suffix}"))


def run_opencli(args: list[str], verbose: bool = False) -> str:
    try:
        result = subprocess.run(
            ["opencli", *args],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("opencli is not installed or not on PATH.") from exc
    if result.returncode != 0:
        raise SystemExit((result.stderr or result.stdout).strip() or "opencli command failed")
    if verbose and result.stdout.strip():
        print(result.stdout, file=sys.stderr)
    return result.stdout


def ask_gemini(prompt: str, wait_seconds: int, new_chat: bool, verbose: bool) -> str:
    args = ["gemini", "ask", "--format", "json", "--timeout", str(wait_seconds)]
    if new_chat:
        args.extend(["--new", "true"])
    if verbose:
        args.append("--verbose")
    args.append(prompt)
    raw = run_opencli(args, verbose=False).strip()
    if verbose and raw:
        print(raw, file=sys.stderr)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse Gemini response JSON: {raw}") from exc

    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict) and "response" in first:
            return str(first["response"])
    if isinstance(parsed, dict) and "response" in parsed:
        return str(parsed["response"])
    raise SystemExit(f"Unexpected Gemini response JSON shape: {raw}")


def build_rectify_prompt(srt_text: str) -> str:
    return (
        "Correct wrong subtitle words based on the overall context of this SRT.\n"
        "Keep every cue index and every timeline line exactly unchanged.\n"
        "Only fix mistaken words in subtitle text.\n"
        "Return only valid SRT, no explanation, no markdown fences.\n\n"
        f"{srt_text}"
    )


def print_labeled_block(label: str, content: str) -> None:
    print(f"=== {label} ===", file=sys.stderr)
    print(content, file=sys.stderr)
    print(file=sys.stderr)


def correct_srt_with_gemini(
    srt_text: str,
    wait_seconds: int,
    new_chat: bool,
    verbose: bool,
) -> str:
    prompt = build_rectify_prompt(srt_text)
    print_labeled_block("SRT INPUT", srt_text)
    print_labeled_block("GEMINI PROMPT", prompt)
    response = ask_gemini(
        prompt=prompt,
        wait_seconds=wait_seconds,
        new_chat=new_chat,
        verbose=verbose,
    )
    candidate = strip_code_fences(response)
    print_labeled_block("GEMINI RESPONSE", candidate)
    return candidate


def rectify_file(args: argparse.Namespace) -> int:
    for raw_srt in args.srt:
        srt_path = Path(raw_srt).expanduser()
        if not srt_path.exists():
            raise SystemExit(f"SRT file not found: {srt_path}")
        if srt_path.suffix.lower() != ".srt":
            raise SystemExit(f"Expected an .srt file: {srt_path}")

        output_path = corrected_srt_path(srt_path)
        print_run_config(
            "RECTIFY CONFIG",
            [
                ("input_srt", srt_path),
                ("output_srt", output_path),
                ("wait_seconds", args.wait_seconds),
                ("new_chat", args.new_chat),
                ("verbose", args.verbose),
            ],
        )

        original_text = srt_path.read_text(encoding="utf-8")
        original_entries = parse_srt_entries(original_text)
        corrected_raw = correct_srt_with_gemini(
            srt_text=original_text,
            wait_seconds=args.wait_seconds,
            new_chat=args.new_chat,
            verbose=args.verbose,
        )
        corrected_entries = parse_srt_entries(corrected_raw)
        if len(corrected_entries) != len(original_entries):
            raise SystemExit(
                "Gemini returned a different number of SRT cues; refusing to change timing layout."
            )

        merged_entries: list[dict[str, str]] = []
        for original, corrected in zip(original_entries, corrected_entries, strict=True):
            merged_entries.append(
                {
                    "index": original["index"],
                    "start": original["start"],
                    "end": original["end"],
                    "text": corrected["text"],
                }
            )

        output_path.write_text(render_srt_entries(merged_entries), encoding="utf-8")
        print(output_path)
    return 0


def extract_sentence_tokens(sentence: Any) -> list[dict[str, float | str]]:
    tokens = getattr(sentence, "tokens", None)
    if not tokens:
        return []

    extracted: list[dict[str, float | str]] = []
    for token in tokens:
        text = getattr(token, "text", "").strip()
        start = getattr(token, "start", None)
        end = getattr(token, "end", None)
        if not text or start is None or end is None:
            continue
        extracted.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "sentence_end": False,
            }
        )
    if extracted:
        extracted[-1]["sentence_end"] = True
    return extracted


def extract_precise_timed_units(result: Any) -> list[dict[str, float | str]]:
    units: list[dict[str, float | str]] = []

    if hasattr(result, "sentences") and result.sentences is not None:
        for sentence in result.sentences:
            sentence_tokens = extract_sentence_tokens(sentence)
            if sentence_tokens:
                units.extend(sentence_tokens)
        if units:
            return units

    if hasattr(result, "segments") and result.segments is not None:
        for segment in result.segments:
            words = segment.get("words") or []
            if words:
                for word in words:
                    text = (word.get("word") or word.get("text") or "").strip()
                    start = word.get("start")
                    end = word.get("end")
                    if not text or start is None or end is None:
                        continue
                    units.append(
                        {
                            "start": float(start),
                            "end": float(end),
                            "text": text,
                            "sentence_end": text_has_sentence_break(text),
                        }
                    )
        return units

    return units


def build_entries_from_segments(result: Any) -> list[dict[str, str]]:
    segments = getattr(result, "segments", None) or []
    entries: list[dict[str, str]] = []
    for index, segment in enumerate(segments, start=1):
        start = segment.get("start")
        end = segment.get("end")
        text = str(segment.get("text", "")).strip()
        if start is None or end is None or not text:
            continue
        entries.append(
            {
                "index": str(index),
                "start": format_srt_timestamp(max(float(start), 0.0)),
                "end": format_srt_timestamp(max(float(end), float(start) + 0.001)),
                "text": text,
            }
        )
    return entries


def extract_aligned_units(alignment_result: Any) -> list[dict[str, float | str]]:
    extracted: list[dict[str, float | str]] = []
    for item in alignment_result:
        text = getattr(item, "text", "").strip()
        start = getattr(item, "start_time", None)
        end = getattr(item, "end_time", None)
        if not text or start is None or end is None:
            continue
        extracted.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "sentence_end": text_has_sentence_break(text),
            }
        )
    return extracted


def align_transcript_words(
    load_model: Any,
    aligner_model_name: str,
    audio_path: str,
    transcript_text: str,
    language: str | None,
    verbose: bool,
) -> list[dict[str, float | str]]:
    if not transcript_text.strip():
        return []

    aligner = load_model(normalize_model(aligner_model_name))
    kwargs: dict[str, Any] = {"text": transcript_text}
    if language is not None:
        kwargs["language"] = language
    alignment_result = aligner.generate(audio_path, verbose=verbose, **kwargs)
    return extract_aligned_units(alignment_result)


def build_srt_cues_from_units(
    units: list[dict[str, float | str]],
    max_chars_per_line: int | None,
) -> list[dict[str, float | str]]:
    cues: list[dict[str, float | str]] = []
    current_units: list[dict[str, float | str]] = []
    current_text = ""

    def flush() -> None:
        nonlocal current_units, current_text
        if not current_units:
            return
        cues.append(
            {
                "start": float(current_units[0]["start"]),
                "end": float(current_units[-1]["end"]),
                "text": current_text,
            }
        )
        current_units = []
        current_text = ""

    for unit in units:
        text = str(unit["text"]).strip()
        if not text:
            continue

        candidate_text = combine_cue_text(current_text, text)
        if (
            max_chars_per_line is not None
            and current_units
            and visible_length(candidate_text) > max_chars_per_line
        ):
            flush()
            candidate_text = text

        current_units.append(unit)
        current_text = candidate_text

        reached_line_limit = (
            max_chars_per_line is not None
            and visible_length(current_text) >= max_chars_per_line
        )
        reached_sentence_end = bool(unit.get("sentence_end")) or text_has_sentence_break(text)
        if reached_line_limit or reached_sentence_end:
            flush()

    flush()
    return cues


def build_exact_srt_cues(
    result: Any,
    load_model: Any | None,
    backend_audio: str,
    max_chars_per_line: int | None,
    aligner_model_name: str,
    language: str | None,
    verbose: bool,
) -> list[dict[str, float | str]]:
    units = extract_precise_timed_units(result)
    if not units and load_model is not None:
        units = align_transcript_words(
            load_model=load_model,
            aligner_model_name=aligner_model_name,
            audio_path=backend_audio,
            transcript_text=getattr(result, "text", ""),
            language=language,
            verbose=verbose,
        )
    if not units:
        return []
    return build_srt_cues_from_units(units, max_chars_per_line)


def write_fcpxml_subtitles(
    result: Any,
    output_prefix: str,
    load_model: Any,
    backend_audio: str,
    max_chars_per_line: int | None,
    aligner_model_name: str,
    language: str | None,
    verbose: bool,
) -> None:
    cues = build_exact_srt_cues(
        result=result,
        load_model=load_model,
        backend_audio=backend_audio,
        max_chars_per_line=max_chars_per_line,
        aligner_model_name=aligner_model_name,
        language=language,
        verbose=verbose,
    )
    if not cues:
        raise SystemExit(
            "FCPXML subtitle export requested, but no exact caption cues were available."
        )

    entries = [
        {
            "index": str(index),
            "start": format_srt_timestamp(max(float(cue["start"]), 0.0)),
            "end": format_srt_timestamp(max(float(cue["end"]), float(cue["start"]) + 0.001)),
            "text": str(cue["text"]).strip(),
        }
        for index, cue in enumerate(cues, start=1)
    ]
    output_path = Path(f"{output_prefix}.fcpxml")
    output_path.write_text(
        render_fcpxml_subtitles(
            entries=entries,
            project_name=output_path.stem,
            language=language,
        ),
        encoding="utf-8",
    )


def rewrite_srt_with_line_limit(
    result: Any,
    output_prefix: str,
    load_model: Any,
    backend_audio: str,
    max_chars_per_line: int,
    aligner_model_name: str,
    language: str | None,
    verbose: bool,
) -> None:
    cues = build_exact_srt_cues(
        result=result,
        load_model=load_model,
        backend_audio=backend_audio,
        max_chars_per_line=max_chars_per_line,
        aligner_model_name=aligner_model_name,
        language=language,
        verbose=verbose,
    )
    if not cues:
        raise SystemExit(
            "Exact SRT splitting requested, but no word-level timestamps were available "
            "and forced alignment did not return aligned words."
        )

    output_path = Path(f"{output_prefix}.srt")
    with output_path.open("w", encoding="utf-8") as handle:
        for index, cue in enumerate(cues, start=1):
            start = max(float(cue["start"]), 0.0)
            end = max(float(cue["end"]), start + 0.001)
            handle.write(f"{index}\n")
            handle.write(
                f"{format_srt_timestamp(start)} --> "
                f"{format_srt_timestamp(end)}\n"
            )
            handle.write(f"{str(cue['text']).strip()}\n\n")


def write_basic_transcription_output(
    result: Any,
    output_prefix: str,
    output_format: str,
    load_model: Any | None,
    backend_audio: str,
    max_chars_per_line: int | None,
    aligner_model_name: str,
    language: str | None,
    verbose: bool,
) -> None:
    output_path = Path(f"{output_prefix}.{output_format}")
    if output_format == "txt":
        output_path.write_text(f"{getattr(result, 'text', '').strip()}\n", encoding="utf-8")
        return
    if output_format == "json":
        payload = getattr(result, "raw", None)
        if payload is None:
            payload = {
                "text": getattr(result, "text", ""),
                "segments": getattr(result, "segments", []),
                "language": getattr(result, "language", None),
            }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if output_format == "srt":
        if max_chars_per_line is not None:
            rewrite_srt_with_line_limit(
                result=result,
                output_prefix=output_prefix,
                load_model=load_model,
                backend_audio=backend_audio,
                max_chars_per_line=max_chars_per_line,
                aligner_model_name=aligner_model_name,
                language=language,
                verbose=verbose,
            )
            return
        output_path.write_text(
            render_srt_entries(build_entries_from_segments(result)),
            encoding="utf-8",
        )
        return
    if output_format == "vtt":
        entries = build_entries_from_segments(result)
        if max_chars_per_line is not None:
            cues = build_exact_srt_cues(
                result=result,
                load_model=load_model,
                backend_audio=backend_audio,
                max_chars_per_line=max_chars_per_line,
                aligner_model_name=aligner_model_name,
                language=language,
                verbose=verbose,
            )
            entries = [
                {
                    "index": str(index),
                    "start": format_srt_timestamp(max(float(cue["start"]), 0.0)),
                    "end": format_srt_timestamp(
                        max(float(cue["end"]), float(cue["start"]) + 0.001)
                    ),
                    "text": str(cue["text"]).strip(),
                }
                for index, cue in enumerate(cues, start=1)
            ]
        output_path.write_text(render_vtt_entries(entries), encoding="utf-8")
        return
    if output_format == "fcpxml":
        write_fcpxml_subtitles(
            result=result,
            output_prefix=output_prefix,
            load_model=load_model,
            backend_audio=backend_audio,
            max_chars_per_line=max_chars_per_line,
            aligner_model_name=aligner_model_name,
            language=language,
            verbose=verbose,
        )
        return
    raise SystemExit(f"Unsupported output format: {output_format}")


def input_requires_ffmpeg(audio_path: Path) -> bool:
    return audio_path.suffix.lower() in FFMPEG_INPUT_SUFFIXES


def decode_with_ffmpeg(audio_path: Path, stack: ExitStack) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            "This input format requires ffmpeg, but ffmpeg was not found on PATH."
        )

    tmp = stack.enter_context(
        tempfile.NamedTemporaryFile(prefix="asr-cli-", suffix=".wav", delete=True)
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"ffmpeg failed to decode input media:\n{result.stderr.strip()}"
        )
    return tmp.name


def transcribe_files(args: argparse.Namespace) -> list[Path]:
    if args.max_chars_per_line is not None and args.max_chars_per_line <= 0:
        raise SystemExit("--max-chars-per-line must be greater than 0")

    resolved_model = normalize_model(args.model)
    load_model: Any | None = None
    generate_transcription = None
    whisper_module = None
    whisper_runtime_model: str | None = None
    if is_official_whisper_model(resolved_model):
        whisper_module = load_official_whisper()
        whisper_runtime_model = official_whisper_runtime_model(resolved_model)
        model = whisper_module.load_model(whisper_runtime_model)
    else:
        load_model, generate_transcription = load_backend()
        model = load_transcription_model(load_model, resolved_model)

    backend_format = "srt" if args.format == "fcpxml" else args.format
    base_kwargs: dict[str, Any] = {
        "format": backend_format,
        "verbose": args.verbose,
    }
    if args.language is not None:
        base_kwargs["language"] = args.language
    if args.chunk_duration is not None:
        base_kwargs["chunk_duration"] = args.chunk_duration
    if args.frame_threshold is not None:
        base_kwargs["frame_threshold"] = args.frame_threshold
    if args.max_tokens is not None:
        base_kwargs["max_tokens"] = args.max_tokens
    if args.stream:
        base_kwargs["stream"] = True
    base_kwargs.update(parse_gen_kwargs(args.gen_kwargs))
    base_kwargs.update(build_mlx_prompt_kwargs(resolved_model, args))

    multiple_inputs = len(args.audio) > 1
    written_paths: list[Path] = []
    for raw_audio in args.audio:
        audio_path = Path(raw_audio).expanduser()
        if not audio_path.exists():
            raise SystemExit(f"Input file not found: {audio_path}")

        output_prefix = choose_output_prefix(
            resolve_output_prefix(audio_path, args.output, multiple_inputs),
            args.format,
        )
        with ExitStack() as stack:
            backend_audio = str(audio_path)
            if input_requires_ffmpeg(audio_path):
                backend_audio = decode_with_ffmpeg(audio_path, stack)

            print_run_config(
                "TRANSCRIBE CONFIG",
                [
                    ("input", audio_path),
                    (
                        "backend",
                        "official-openai-whisper"
                        if whisper_module is not None
                        else "mlx-audio",
                    ),
                    ("model", args.model),
                    ("resolved_model", resolved_model),
                    ("backend_model", whisper_runtime_model or resolved_model),
                    ("output", Path(f"{output_prefix}.{args.format}")),
                    ("format", args.format),
                    ("language", args.language),
                    ("context", args.context),
                    ("prompt", args.prompt),
                    ("spelling_prompt", getattr(args, "spelling_prompt", None)),
                    ("chunk_duration", args.chunk_duration),
                    ("frame_threshold", args.frame_threshold),
                    ("max_tokens", args.max_tokens),
                    ("max_chars_per_line", args.max_chars_per_line),
                    ("aligner_model", args.aligner_model),
                    ("stream", args.stream),
                    ("gen_kwargs", args.gen_kwargs),
                    ("new_chat", getattr(args, "new_chat", None)),
                    ("verbose", args.verbose),
                    ("decoded_with_ffmpeg", backend_audio != str(audio_path)),
                ],
            )

            if whisper_module is not None:
                result = transcribe_with_official_whisper(
                    whisper_module=whisper_module,
                    model=model,
                    audio_path=backend_audio,
                    args=args,
                )
                write_basic_transcription_output(
                    result=result,
                    output_prefix=output_prefix,
                    load_model=load_model,
                    backend_audio=backend_audio,
                    output_format=args.format,
                    max_chars_per_line=args.max_chars_per_line,
                    aligner_model_name=args.aligner_model,
                    language=args.language,
                    verbose=args.verbose,
                )
            else:
                assert generate_transcription is not None
                result = generate_transcription(
                    model=model,
                    audio=backend_audio,
                    output_path=output_prefix,
                    **base_kwargs,
                )
                if args.format in {"txt", "json", "srt", "vtt", "fcpxml"}:
                    write_basic_transcription_output(
                        result=result,
                        output_prefix=output_prefix,
                        load_model=load_model,
                        backend_audio=backend_audio,
                        output_format=args.format,
                        max_chars_per_line=args.max_chars_per_line,
                        aligner_model_name=args.aligner_model,
                        language=args.language,
                        verbose=args.verbose,
                    )
                    if args.format == "fcpxml":
                        intermediate_srt = Path(f"{output_prefix}.srt")
                        if intermediate_srt.exists():
                            intermediate_srt.unlink()
        written_paths.append(Path(f"{output_prefix}.{args.format}"))
        if not args.verbose and hasattr(result, "text"):
            print(result.text)

    return written_paths


def run_all(args: argparse.Namespace) -> int:
    transcribe_args = argparse.Namespace(
        action="transcribe",
        audio=[args.input],
        model=args.model,
        resolved_model=False,
        list_models=False,
        output=args.output,
        format="srt",
        language=args.language,
        context=args.context,
        prompt=args.prompt,
        spelling_prompt=args.spelling_prompt,
        chunk_duration=args.chunk_duration,
        frame_threshold=args.frame_threshold,
        max_tokens=args.max_tokens,
        max_chars_per_line=args.max_chars_per_line,
        aligner_model=args.aligner_model,
        stream=args.stream,
        gen_kwargs=args.gen_kwargs,
        verbose=args.verbose,
    )
    written_paths = transcribe_files(transcribe_args)
    if len(written_paths) != 1:
        raise SystemExit("The all action expected exactly one generated SRT file.")

    rectify_args = argparse.Namespace(
        action="rectify",
        srt=[str(written_paths[0])],
        wait_seconds=args.wait_seconds,
        new_chat=args.new_chat,
        verbose=args.verbose,
    )
    return rectify_file(rectify_args)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.action == "rectify":
        return rectify_file(args)
    if args.action == "all":
        return run_all(args)

    if args.list_models:
        print_model_aliases()
        return 0

    resolved_model = normalize_model(args.model)
    if args.resolved_model:
        print(resolved_model)
        return 0

    transcribe_files(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
