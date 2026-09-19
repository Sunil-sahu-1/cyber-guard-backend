"""
Voice deepfake / synthetic-speech verification engine.

This is an explainable acoustic baseline. It does NOT assume that an
AI voice has one constant frequency. Instead it combines multiple
time-varying speech signals: F0/pitch variation, spectral variation,
MFCC dynamics, energy/prosody variation, zero-crossing behaviour,
spectral flatness, silence structure, clipping and other acoustic
signals.

For production-grade detection, this module is designed to be replaced
or fused with a model trained on ASVspoof-style bona-fide/spoof data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess

import numpy as np


MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_DURATION_SECONDS = 120.0
TARGET_SAMPLE_RATE = 16_000

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".webm",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".caf",
    ".amr",
    ".mka",
    ".ac3",
    ".mp2",
    ".mpeg",
    ".mpga",
}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return round(max(low, min(high, float(value))), 2)


def _safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else default


def _safe_std(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if values.size else default


def _safe_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else default


def _variation_score(value: float, low: float, high: float) -> float:
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    return (high - value) / (high - low)


def _load_audio(file_path: str):
    """Decode audio with the bundled FFmpeg executable for broad codec support."""
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("Audio decoder is not installed. Install imageio-ffmpeg.") from error

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        file_path,
        "-map",
        "0:a:0",
        "-t",
        str(MAX_DURATION_SECONDS),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=MAX_DURATION_SECONDS + 20,
        )
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg audio decoder is unavailable.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Audio decoding timed out.") from error
    except subprocess.CalledProcessError as error:
        decoder_error = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(decoder_error or "The uploaded audio format or codec could not be decoded.") from error

    audio = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError("No audio samples were found. Make sure the file contains an audio track.")

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak

    return audio, TARGET_SAMPLE_RATE

def _feature_extraction(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    import librosa

    n_fft = 1024
    hop_length = 256

    rms = librosa.feature.rms(
        y=audio,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    mfcc_delta = librosa.feature.delta(mfcc)

    f0 = librosa.yin(
        audio,
        fmin=70,
        fmax=400,
        sr=sample_rate,
        frame_length=2048,
        hop_length=hop_length,
    )

    f0 = np.asarray(f0, dtype=float)
    voiced_mask = np.isfinite(f0) & (f0 >= 70) & (f0 <= 400)
    voiced_f0 = f0[voiced_mask]

    non_silent = librosa.effects.split(
        audio,
        top_db=35,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    speech_samples = sum(
        max(0, int(end) - int(start))
        for start, end in non_silent
    )

    duration = len(audio) / sample_rate
    speech_ratio = (
        min(1.0, speech_samples / len(audio))
        if len(audio)
        else 0.0
    )

    silence_ratio = 1.0 - speech_ratio

    rms_db = librosa.amplitude_to_db(
        np.maximum(rms, 1e-7),
        ref=1.0,
    )

    mfcc_frame_variability = float(
        np.mean(
            np.std(
                mfcc[1:],
                axis=1,
            )
        )
    )

    mfcc_delta_variability = float(
        np.mean(
            np.std(
                mfcc_delta,
                axis=1,
            )
        )
    )

    spectral_flux = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )

    harmonic, percussive = librosa.effects.hpss(audio)
    harmonic_energy = float(np.mean(harmonic ** 2))
    percussive_energy = float(np.mean(percussive ** 2))
    harmonic_ratio = harmonic_energy / max(
        harmonic_energy + percussive_energy,
        1e-9,
    )

    peak = float(np.max(np.abs(audio)))
    clipping_ratio = float(
        np.mean(np.abs(audio) >= 0.995)
    )

    if voiced_f0.size:
        pitch_mean = _safe_mean(voiced_f0)
        pitch_std = _safe_std(voiced_f0)
        pitch_range = (
            _safe_percentile(voiced_f0, 95)
            - _safe_percentile(voiced_f0, 5)
        )
        pitch_cv = pitch_std / max(pitch_mean, 1.0)

        periods = 1.0 / np.maximum(voiced_f0, 1e-6)
        jitter = (
            _safe_mean(np.abs(np.diff(periods)))
            / max(_safe_mean(periods), 1e-9)
        )
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_range = 0.0
        pitch_cv = 0.0
        jitter = 0.0

    voiced_rms_length = min(
        len(rms),
        len(voiced_mask),
    )
    voiced_rms = rms[:voiced_rms_length][
        voiced_mask[:voiced_rms_length]
    ]

    shimmer = (
        _safe_mean(np.abs(np.diff(voiced_rms)))
        / max(_safe_mean(voiced_rms), 1e-9)
        if voiced_rms.size > 1
        else 0.0
    )

    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "speech_ratio": round(speech_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
        "pitch_mean_hz": round(pitch_mean, 2),
        "pitch_std_hz": round(pitch_std, 2),
        "pitch_range_hz": round(pitch_range, 2),
        "pitch_variation_cv": round(pitch_cv, 4),
        "jitter": round(jitter, 5),
        "shimmer": round(shimmer, 5),
        "spectral_centroid_mean_hz": round(_safe_mean(centroid), 2),
        "spectral_centroid_std_hz": round(_safe_std(centroid), 2),
        "spectral_bandwidth_std_hz": round(_safe_std(bandwidth), 2),
        "spectral_flatness_mean": round(_safe_mean(flatness), 5),
        "spectral_flatness_std": round(_safe_std(flatness), 5),
        "mfcc_variability": round(mfcc_frame_variability, 4),
        "mfcc_delta_variability": round(mfcc_delta_variability, 4),
        "energy_std_db": round(_safe_std(rms_db), 2),
        "energy_range_db": round(
            _safe_percentile(rms_db, 95)
            - _safe_percentile(rms_db, 5),
            2,
        ),
        "zero_crossing_std": round(_safe_std(zcr), 5),
        "spectral_flux_std": round(_safe_std(spectral_flux), 4),
        "harmonic_ratio": round(harmonic_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 6),
        "voiced_frame_ratio": round(
            float(np.mean(voiced_mask)),
            4,
        ),
    }


def _score_features(features: dict[str, Any]) -> tuple[float, list[str], dict[str, float]]:
    indicators: list[str] = []
    components: dict[str, float] = {}

    pitch_cv = float(features["pitch_variation_cv"])
    centroid_std = float(features["spectral_centroid_std_hz"])
    bandwidth_std = float(features["spectral_bandwidth_std_hz"])
    mfcc_var = float(features["mfcc_variability"])
    mfcc_delta = float(features["mfcc_delta_variability"])
    energy_std = float(features["energy_std_db"])
    zcr_std = float(features["zero_crossing_std"])
    flatness_std = float(features["spectral_flatness_std"])
    flux_std = float(features["spectral_flux_std"])
    speech_ratio = float(features["speech_ratio"])
    clipping_ratio = float(features["clipping_ratio"])
    jitter = float(features["jitter"])
    shimmer = float(features["shimmer"])

    # These are weak signals. A human can naturally have stable pitch,
    # and an AI voice can deliberately introduce variation.
    pitch_signal = _variation_score(pitch_cv, 0.025, 0.12)
    spectral_signal = (
        _variation_score(centroid_std, 120.0, 650.0)
        + _variation_score(bandwidth_std, 100.0, 700.0)
    ) / 2.0
    mfcc_signal = (
        _variation_score(mfcc_var, 4.0, 20.0)
        + _variation_score(mfcc_delta, 1.0, 8.0)
    ) / 2.0
    energy_signal = _variation_score(energy_std, 2.0, 8.0)
    jitter_signal = _variation_score(jitter, 0.002, 0.02)
    shimmer_signal = _variation_score(shimmer, 0.01, 0.08)

    temporal_signal = (
        _variation_score(zcr_std, 0.005, 0.025)
        + _variation_score(flatness_std, 0.004, 0.025)
        + _variation_score(flux_std, 0.02, 0.20)
    ) / 3.0

    components["pitch_variation"] = round(pitch_signal * 100, 2)
    components["spectral_variation"] = round(spectral_signal * 100, 2)
    components["mfcc_dynamics"] = round(mfcc_signal * 100, 2)
    components["energy_dynamics"] = round(energy_signal * 100, 2)
    components["temporal_texture"] = round(temporal_signal * 100, 2)
    components["jitter"] = round(jitter_signal * 100, 2)
    components["shimmer"] = round(shimmer_signal * 100, 2)

    score = (
        pitch_signal * 20
        + spectral_signal * 20
        + mfcc_signal * 18
        + energy_signal * 14
        + temporal_signal * 12
        + jitter_signal * 8
        + shimmer_signal * 8
    )

    if speech_ratio < 0.15:
        score = min(100.0, score + 8.0)
        indicators.append(
            "Very little speech activity was detected; confidence is reduced."
        )

    if clipping_ratio > 0.02:
        score = min(100.0, score + 4.0)
        indicators.append(
            "Significant clipping was detected; recording quality may affect verification."
        )

    if pitch_signal >= 0.75:
        indicators.append(
            "Pitch variation is unusually low across voiced segments."
        )

    if spectral_signal >= 0.75:
        indicators.append(
            "Spectral characteristics remain unusually consistent across the recording."
        )

    if mfcc_signal >= 0.75:
        indicators.append(
            "Speech timbre features show unusually low frame-to-frame variation."
        )

    if energy_signal >= 0.75:
        indicators.append(
            "Energy/prosody variation is unusually smooth."
        )

    if temporal_signal >= 0.75:
        indicators.append(
            "Temporal acoustic texture shows low natural variation."
        )

    if jitter_signal >= 0.75:
        indicators.append(
            "Micro-pitch variation (jitter) is unusually low."
        )

    if shimmer_signal >= 0.75:
        indicators.append(
            "Micro-amplitude variation (shimmer) is unusually low."
        )

    # Repeated weak signals are more meaningful than any single feature.
    if sum(
        value >= 75.0
        for value in components.values()
    ) >= 3:
        indicators.append(
            "Several independent acoustic signals are consistent with synthetic or heavily processed speech."
        )

    return _clamp(score), indicators, components


def _prediction(score: float) -> str:
    if score >= 80:
        return "LIKELY_AI_GENERATED"
    if score >= 60:
        return "HIGH_SYNTHETIC_VOICE_RISK"
    if score >= 40:
        return "SUSPICIOUS_VOICE"
    if score >= 20:
        return "LOW_SYNTHETIC_VOICE_RISK"
    return "LIKELY_HUMAN_VOICE"


def _severity(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    if score >= 20:
        return "LOW"
    return "SAFE"


def _confidence(score: float, features: dict[str, Any]) -> float:
    duration = float(features["duration_seconds"])
    speech_ratio = float(features["speech_ratio"])

    quality_factor = min(1.0, max(0.0, duration / 8.0))
    speech_factor = min(1.0, speech_ratio / 0.55)

    base = 0.55 + (quality_factor * 0.18) + (speech_factor * 0.17)

    if duration < 3.0:
        base -= 0.08

    return round(
        min(0.90, max(0.50, base)),
        2,
    )


def analyze_voice_file(
    file_path: str,
    file_name: str,
    file_size: int,
) -> dict[str, Any]:
    extension = Path(file_name).suffix.lower()

    if not file_name:
        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Audio file name is required.",
        }

    if file_size <= 0:
        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Audio file is empty.",
        }

    if file_size > MAX_FILE_SIZE:
        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Audio file exceeds the 20 MB limit.",
        }

    if extension not in AUDIO_EXTENSIONS:
        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Unsupported audio format.",
        }

    try:
        audio, sample_rate = _load_audio(file_path)
        features = _feature_extraction(audio, sample_rate)
        score, indicators, components = _score_features(features)
    except Exception as error:
        return {
            "is_valid": False,
            "prediction": "AUDIO_READ_ERROR",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": f"Audio could not be decoded: {error}",
        }

    if features["duration_seconds"] < 1.0:
        return {
            "is_valid": False,
            "prediction": "AUDIO_TOO_SHORT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Please provide at least 1 second of speech.",
        }

    score = _clamp(score)

    if not indicators:
        indicators.append(
            "No strong synthetic-speech acoustic pattern was detected by the current baseline."
        )

    recommendation = {
        "CRITICAL": "Do not rely on the voice alone. Verify the speaker through an independent channel.",
        "HIGH": "Treat the voice as suspicious and perform an independent identity check.",
        "MEDIUM": "Request additional verification before trusting the speaker.",
        "LOW": "The recording contains weak synthetic-speech signals; continue monitoring.",
        "SAFE": "No strong synthetic-speech pattern was detected, but this is not proof of human origin.",
    }[_severity(score)]

    return {
        "analysis_type": "voice_anti_spoofing_acoustic_baseline",
        "is_valid": True,
        "risk_score": score,
        "severity": _severity(score),
        "prediction": _prediction(score),
        "confidence": _confidence(score, features),
        "indicators": indicators,
        "features": features,
        "signal_components": components,
        "recommendation": recommendation,
        "detector_note": (
            "This baseline uses multiple acoustic signals. "
            "Frequency consistency alone is not sufficient to identify AI-generated speech."
        ),
    }
