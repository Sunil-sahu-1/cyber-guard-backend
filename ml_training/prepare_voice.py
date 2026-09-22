"""Prepare a small local voice deepfake dataset for Cyber Guard.

Streams MLAAD-tiny, keeps 250 bona-fide and 250 spoof examples, trims each
clip to at most 5 seconds, writes 16 kHz mono WAV files, and creates
datasets/voice/voice_manifest.csv.

The MLAAD-tiny soundfolder does not expose a "label" column in the streamed
rows, so this script detects the class from available metadata/path fields.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "datasets" / "voice"
REAL = VOICE / "real"
FAKE = VOICE / "fake"

DATASET_ID = "mueller91/MLAAD-tiny"
PER_CLASS = 250
MAX_SECONDS = 5
TARGET_SR = 16_000


def detect_label(item: dict) -> int | None:
    """Return 0 for bona-fide/real and 1 for spoof/fake, if detectable."""
    # Some dataset variants expose an explicit class-like field.
    for key in ("label", "class", "category", "target", "type"):
        if key in item:
            value = item[key]
            if isinstance(value, (int, np.integer)):
                if int(value) in (0, 1):
                    return int(value)
            text = str(value).strip().lower()
            if text in {"0", "bonafide", "bona-fide", "real", "human"}:
                return 0
            if text in {"1", "spoof", "fake", "synthetic", "ai"}:
                return 1

    # MLAAD-tiny is a soundfolder dataset; class information is represented
    # by the source path/folder rather than a label column.
    audio = item.get("audio")
    candidates = []
    if isinstance(audio, dict):
        candidates.extend(
            [audio.get("path"), audio.get("filename"), audio.get("name")]
        )

    for key in ("path", "file", "filename", "file_name", "name"):
        if key in item:
            candidates.append(item[key])

    haystack = " ".join(str(x) for x in candidates if x).lower()
    if any(token in haystack for token in ("bona-fide", "bonafide", "bona_fide")):
        return 0
    if any(token in haystack for token in ("spoof", "fake", "synthetic")):
        return 1

    return None


def main() -> None:
    REAL.mkdir(parents=True, exist_ok=True)
    FAKE.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    real_count = 0
    fake_count = 0

    print(f"[voice] source: {DATASET_ID}")
    print("[voice] target: 250 real + 250 spoof")
    print("[voice] clip length: <= 5 seconds, 16 kHz mono")

    for item in ds:
        label = detect_label(item)

        if label is None:
            continue
        if label == 0 and real_count >= PER_CLASS:
            continue
        if label == 1 and fake_count >= PER_CLASS:
            continue

        audio_obj = item.get("audio")
        if not isinstance(audio_obj, dict):
            continue

        audio = np.asarray(audio_obj["array"], dtype=np.float32)
        sample_rate = int(audio_obj["sampling_rate"])

        if audio.ndim > 1:
            # Handle either (samples, channels) or (channels, samples).
            if audio.shape[0] < audio.shape[1]:
                audio = np.mean(audio, axis=0)
            else:
                audio = np.mean(audio, axis=1)

        if sample_rate != TARGET_SR:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=TARGET_SR
            )
            sample_rate = TARGET_SR

        audio = audio[: TARGET_SR * MAX_SECONDS]

        if label == 0:
            path = REAL / f"{real_count:04d}.wav"
            real_count += 1
            normalized_label = 0
        else:
            path = FAKE / f"{fake_count:04d}.wav"
            fake_count += 1
            normalized_label = 1

        sf.write(path, audio, sample_rate, subtype="PCM_16")

        if (real_count + fake_count) % 50 == 0:
            print(f"[voice] {real_count}/250 real, {fake_count}/250 spoof")

        if real_count >= PER_CLASS and fake_count >= PER_CLASS:
            break

    if real_count < PER_CLASS or fake_count < PER_CLASS:
        raise SystemExit(
            f"Could not collect 500 balanced samples: "
            f"{real_count} real, {fake_count} spoof"
        )

    manifest = VOICE / "voice_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])

        for path in sorted(REAL.glob("*.wav")):
            writer.writerow([path.relative_to(ROOT).as_posix(), 0])

        for path in sorted(FAKE.glob("*.wav")):
            writer.writerow([path.relative_to(ROOT).as_posix(), 1])

    total_bytes = sum(p.stat().st_size for p in VOICE.rglob("*.wav"))
    print(f"[done] {real_count} real + {fake_count} spoof")
    print(f"[done] audio size: {total_bytes / (1024 * 1024):.1f} MB")
    print(f"[done] manifest: {manifest}")


if __name__ == "__main__":
    main()
