from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


MODEL_SIZES = ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
FORMAT_CHOICES = ("txt", "json", "srt", "vtt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr-cli",
        description="ASR CLI for Windows using Faster-Whisper",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio or video to text/subtitles")
    transcribe_parser.add_argument("input", help="Audio or video file to transcribe")
    transcribe_parser.add_argument(
        "--model",
        "-m",
        default="small",
        choices=MODEL_SIZES,
        help="Whisper model size (default: small)",
    )
    transcribe_parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Source language (e.g., en, zh, ja). Auto-detect if not specified.",
    )
    transcribe_parser.add_argument(
        "--format",
        "-f",
        default="txt",
        choices=FORMAT_CHOICES,
        help="Output format (default: txt)",
    )
    transcribe_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: same as input with new extension)",
    )
    transcribe_parser.add_argument(
        "--max-chars-per-line",
        type=int,
        default=None,
        help="Max characters per line in SRT/VTT subtitles",
    )
    transcribe_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress info",
    )
    transcribe_parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate transcript to English (via Gemini, requires opencli and VPN)",
    )
    transcribe_parser.add_argument(
        "--translate-api-key",
        default=None,
        help="Not used anymore (kept for compatibility)",
    )
    return parser


def format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def parse_srt_timestamp(value: str) -> float:
    hours, minutes, rest = value.replace(",", ":").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def format_vtt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


SENTENCE_BREAK_CHARS = {"。", "！", "？", "!", "?", "."}


def text_has_sentence_break(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped[-1] in SENTENCE_BREAK_CHARS


def visible_length(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def combine_cue_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1].isalnum() and right[0].isalnum():
        return f"{left} {right}"
    return f"{left}{right}"


def build_srt_cues(
    segments: list[dict[str, Any]],
    max_chars_per_line: int | None,
) -> list[dict[str, str]]:
    if not segments:
        return []

    cues = []
    for i, seg in enumerate(segments, 1):
        cues.append({
            "index": str(i),
            "start": format_srt_timestamp(seg["start"]),
            "end": format_srt_timestamp(seg["end"]),
            "text": seg["text"].strip(),
        })

    if max_chars_per_line is None:
        return cues

    units = []
    for seg in segments:
        units.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "sentence_end": text_has_sentence_break(seg["text"]),
        })

    new_cues = []
    current_units = []
    current_text = ""

    def flush():
        nonlocal current_units, current_text
        if not current_units:
            return
        new_cues.append({
            "index": str(len(new_cues) + 1),
            "start": format_srt_timestamp(current_units[0]["start"]),
            "end": format_srt_timestamp(current_units[-1]["end"]),
            "text": current_text,
        })
        current_units = []
        current_text = ""

    for unit in units:
        text = unit["text"]
        if not text:
            continue

        candidate_text = combine_cue_text(current_text, text)
        if current_units and visible_length(candidate_text) > max_chars_per_line:
            flush()
            candidate_text = text

        current_units.append(unit)
        current_text = candidate_text

        reached_limit = visible_length(current_text) >= max_chars_per_line
        reached_sentence = unit.get("sentence_end", False) or text_has_sentence_break(text)
        if reached_limit or reached_sentence:
            flush()

    flush()
    return new_cues


def render_srt(cues: list[dict[str, str]]) -> str:
    blocks = []
    for cue in cues:
        blocks.append(f"{cue['index']}\n{cue['start']} --> {cue['end']}\n{cue['text']}")
    return "\n\n".join(blocks) + "\n"


def render_vtt(cues: list[dict[str, str]]) -> str:
    blocks = ["WEBVTT"]
    for cue in cues:
        start = format_vtt_timestamp(parse_srt_timestamp(cue["start"]))
        end = format_vtt_timestamp(parse_srt_timestamp(cue["end"]))
        blocks.append(f"{start} --> {end}\n{cue['text']}")
    return "\n\n".join(blocks) + "\n"


def translate_cues_via_cli(cues: list[dict[str, str]], verbose: bool = False) -> list[dict[str, str]]:
    import subprocess
    import json as _json

    OPENCLI_CMD = "e:\\nvm4w\\nodejs\\opencli.cmd"
    BATCH_SIZE = 10

    translated_cues = []
    total = len(cues)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        if verbose:
            print(f"  Translating cues {batch_start + 1}-{batch_end} of {total}...", file=sys.stderr)

        batch = cues[batch_start:batch_end]
        chinese_text = "\n".join(f"[{i + 1}] {cue['text']}" for i, cue in enumerate(batch))
        prompt = "翻译成英文，只输出编号和翻译，格式为[编号] 英文，不要其他内容：\n" + chinese_text

        cmd = f'{OPENCLI_CMD} gemini ask {json.dumps(prompt)} -f json'
        result = subprocess.run(
            cmd,
            shell=True, capture_output=True, text=True, encoding="utf-8", timeout=180
        )

        try:
            response = _json.loads(result.stdout)
            translated_text = response[0]["response"]
            if translated_text.startswith("💬 "):
                translated_text = translated_text[2:]

            import re
            parts = re.split(r'(?=\[\d+\])', translated_text.strip())
            parts = [p.strip() for p in parts if p.strip()]

            for i, cue in enumerate(batch):
                if i < len(parts):
                    text_en = parts[i]
                    if "]" in text_en:
                        text_en = text_en.split("]", 1)[-1].strip()
                else:
                    text_en = cue["text"]
                translated_cues.append({
                    "index": str(len(translated_cues) + 1),
                    "start": cue["start"],
                    "end": cue["end"],
                    "text": text_en,
                })
        except Exception as e:
            if verbose:
                print(f"  Batch failed: {e}, keeping original", file=sys.stderr)
            for cue in batch:
                translated_cues.append({
                    "index": str(len(translated_cues) + 1),
                    "start": cue["start"],
                    "end": cue["end"],
                    "text": cue["text"],
                })

    return translated_cues


def transcribe_file(args: argparse.Namespace) -> int:
    from faster_whisper import WhisperModel

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    model_path = "Systran/faster-whisper-tiny"
    if args.model == "small":
        model_path = "Systran/faster-whisper-small"
    elif args.model == "medium":
        model_path = "Systran/faster-whisper-medium"
    elif args.model == "large":
        model_path = "Systran/faster-whisper-large"

    if args.verbose:
        print(f"Loading model: {model_path}", file=sys.stderr)

    model = WhisperModel(model_path)

    if args.verbose:
        print(f"Transcribing: {input_path}", file=sys.stderr)

    word_timestamps = args.format in {"srt", "vtt"} or args.max_chars_per_line is not None
    segments, info = model.transcribe(
        str(input_path),
        language=args.language,
        word_timestamps=word_timestamps,
    )

    if args.verbose:
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})", file=sys.stderr)

    seg_list = list(segments)

    if args.format == "txt":
        text = " ".join(seg.text.strip() for seg in seg_list)
        if args.translate:
            if args.verbose:
                print("Translating text...", file=sys.stderr)
            text = translate_cues_via_cli([{"index": "1", "start": "0", "end": "0", "text": text}], args.verbose)[0]["text"]
        output_text = f"{text}\n"
    elif args.format == "json":
        output_data = {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in seg_list
            ],
        }
        if args.translate:
            if args.verbose:
                print(f"Translating {len(seg_list)} segments...", file=sys.stderr)
            seg_dicts = [{"start": "0", "end": "0", "text": seg.text} for seg in seg_list]
            translated = translate_cues_via_cli(seg_dicts, args.verbose)
            for i, seg in enumerate(output_data["segments"]):
                seg["text_en"] = translated[i]["text"]
        output_text = json.dumps(output_data, ensure_ascii=False, indent=2) + "\n"
    elif args.format in {"srt", "vtt"}:
        seg_dicts = []
        for seg in seg_list:
            seg_dicts.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })
        cues = build_srt_cues(seg_dicts, args.max_chars_per_line)
        if args.translate:
            if args.verbose:
                print(f"Translating {len(cues)} cues to English...", file=sys.stderr)
            cues = translate_cues_via_cli(cues, args.verbose)
        if args.format == "srt":
            output_text = render_srt(cues)
        else:
            output_text = render_vtt(cues)
    else:
        print(f"Error: Unknown format: {args.format}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        output_path = input_path.with_suffix(f".{args.format}")

    output_path.write_text(output_text, encoding="utf-8")
    print(f"Output written to: {output_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.action == "transcribe":
        return transcribe_file(args)

    print(f"Unknown action: {args.action}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())