"""Prepare a small local voice deepfake dataset for Cyber Guard.

Downloads examples through Hugging Face streaming, keeps 250 bona-fide and
250 spoof examples, trims each clip to at most 5 seconds, writes 16 kHz mono
WAV files, and creates datasets/voice/voice_manifest.csv.

The source dataset is not committed to GitHub. Only the prepared local files
are used by the lightweight voice training script.
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
        label = int(item["label"])

        if label == 0 and real_count >= PER_CLASS:
            continue
        if label == 1 and fake_count >= PER_CLASS:
            continue
        if label not in (0, 1):
            continue

        audio = np.asarray(item["audio"]["array"], dtype=np.float32)
        sample_rate = int(item["audio"]["sampling_rate"])

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample only when necessary. librosa is already a project dependency.
        if sample_rate != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SR)
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
